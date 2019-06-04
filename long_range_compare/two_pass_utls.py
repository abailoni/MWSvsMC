from .load_datasets import get_dataset_data, get_dataset_offsets, CREMI_crop_slices, CREMI_sub_crops_slices
import vigra
import numpy as np

from segmfriends.algorithms.agglo import GreedyEdgeContractionAgglomeraterFromSuperpixels, GreedyEdgeContractionAgglomerater

from concurrent import futures
from itertools import product

import nifty
import nifty.graph.rag as nrag



def GUACA_agglomerator(affs, offsets, previous_segmentation=None,
                       previous_edges = None, previous_weights = None,
                       return_state = False,
    **agglo_parameters):

    # update_rule = parameters.get(["extra_aggl_kwargs"], {}).get("update_rule", "mean")
    # nb_threads = parameters.get('n_threads']

    if previous_segmentation is None:
        segm_pipeline = GreedyEdgeContractionAgglomerater(offsets=offsets,
                                                          **agglo_parameters)
        pred_segm, out_dict = segm_pipeline(affs)
    else:
        segm_pipeline = GreedyEdgeContractionAgglomeraterFromSuperpixels(
        offsets,
        **agglo_parameters)

        # Replace zero-label in given segmentation with pixel-labels:
        max_label = previous_segmentation.max()
        pixel_segm = np.arange(np.prod(previous_segmentation.shape), dtype='uint64').reshape(previous_segmentation.shape) + max_label

        new_prev_segm = np.where(previous_segmentation == 0, pixel_segm, previous_segmentation)

        # Make prediction:
        pred_segm, out_dict = segm_pipeline(affs, new_prev_segm,
                                     previous_uv_ids=previous_edges,
                                     previous_edge_weights=previous_weights[:,0],
                                     previous_edge_sizes=previous_weights[:,1])


    if return_state:
        data = out_dict['edge_data_contracted_graph']
        uv_ids = data[:,:2].astype('uint64')
        edge_data = data[:,2:]
        return pred_segm, (uv_ids, edge_data)
    else:
        return pred_segm




def mask_corners(input_, halo):
    ndim = input_.ndim
    shape = input_.shape

    corners = ndim * [[0, 1]]
    corners = product(*corners)

    for corner in corners:
        corner_bb = tuple(slice(0, ha) if co == 0 else slice(sh - ha, sh)
                          for ha, co, sh in zip(halo, shape, corner))
        input_[corner_bb] = 0

    return input_




def make_checkorboard(blocking):
    """
    """
    blocks1 = [0]
    blocks2 = []
    all_blocks = [0]

    def recurse(current_block, insert_list):
        other_list = blocks1 if insert_list is blocks2 else blocks2
        for dim in range(3):
            ngb_id = blocking.getNeighborId(current_block, dim, False)
            if ngb_id != -1:
                if ngb_id not in all_blocks:
                    insert_list.append(ngb_id)
                    all_blocks.append(ngb_id)
                    recurse(ngb_id, other_list)

    recurse(0, blocks2)
    all_blocks = blocks1 + blocks2
    expected = set(range(blocking.numberOfBlocks))
    assert len(all_blocks) == len(expected), "%i, %i" % (len(all_blocks), len(expected))
    assert len(set(all_blocks) - expected) == 0
    assert len(blocks1) == len(blocks2), "%i, %i" % (len(blocks1), len(blocks2))
    return blocks1, blocks2


# find segments in segmentation that originate from seeds
def get_assignments(segmentation, seeds):
    seed_ids, seed_indices = np.unique(seeds, return_index=True)
    # 0 stands for unseeded
    seed_ids, seed_indices = seed_ids[1:], seed_indices[1:]
    seg_ids = segmentation.ravel()[seed_indices]
    assignments = np.concatenate([seed_ids[:, None], seg_ids[:, None]], axis=1).astype(np.int64)
    return assignments


def two_pass_agglomeration(affinities, offsets, agglomerator,
                           block_shape, halo, n_threads):
    """ Run two-pass agglommeration
    """
    assert affinities.ndim == 4
    assert affinities.shape[0] == len(offsets)
    assert callable(agglomerator)
    assert len(block_shape) == len(halo) == 3

    shape = affinities.shape[1:]
    blocking = nifty.tools.blocking([0, 0, 0], list(shape), list(block_shape))
    block_size = np.prod(block_shape)

    segmentation = np.zeros(shape, dtype='uint64')

    # calculations for pass 1:
    #
    def pass1(block_id):
        # TODO we could already add some halo here, that might help to make results more consistent

        # load the affinities from the current block
        block = blocking.getBlock(block_id)
        bb = tuple(slice(beg, end) for beg, end in zip(block.begin, block.end))
        aff_bb = (slice(None),) + bb
        # mutex watershed changes the affs, so we need to copy here
        affs = affinities[aff_bb].copy()

        # get the segmentation and state from our agglomeration function
        seg, state = agglomerator(affs, offsets, return_state=True)

        # offset the segmentation with the lowest block coordinate to
        # make segmentation ids unique
        # id_offset = block_id * block_size
        # seg += id_offset
        # uvs += id_offset

        # write out the segmentation
        # segmentation[bb] = seg

        uvs, weights = state
        return uvs, weights, seg, bb

    # get blocks corresponding to the two checkerboard colorings
    blocks1, blocks2 = make_checkorboard(blocking)

    with futures.ThreadPoolExecutor(n_threads) as tp:
        tasks = [tp.submit(pass1, block_id) for block_id in blocks1]
        results = [t.result() for t in tasks]
    # results = [pass1(block_id) for block_id in blocks1]

    # Combine segmentations:
    # add offset to each of them with the maximum segm values to
    # make segmentation ids unique
    id_offset = 0
    uvs_collected = []
    for res in results:
        seg, bb = res[2:]
        # Write out segmentation:
        segmentation[bb] = seg + id_offset
        # Update uv IDs:
        uvs_collected.append(res[0] + id_offset)
        id_offset += seg.max()

    # combine results and build graph corresponding to it
    uvs = np.concatenate(uvs_collected, axis=0)
    n_labels = int(uvs.max()) + 1
    graph = nifty.graph.undirectedGraph(n_labels)
    graph.insertEdges(uvs)
    weights = np.concatenate([res[1] for res in results], axis=0)
    assert len(uvs) == len(weights)

    # calculations for pass 2:
    #
    def pass2(block_id):
        # load segmentation from pass1 from the current block with halo
        block = blocking.getBlockWithHalo(block_id, list(halo))
        bb = tuple(slice(beg, end) for beg, end in zip(block.outerBlock.begin, block.outerBlock.end))
        seg = segmentation[bb]
        # mask the corners, because these are not part of the seeds, and could already be written by path 2
        # seg = mask_corners(seg, halo)

        # load affinties
        aff_bb = (slice(None),) + bb
        # mutex watershed changes the affs, so we need to copy here
        affs = affinities[aff_bb].copy()

        # get the state of the segmentation from pass 1
        # TODO maybe there is a better option than doing this with the rag
        rag = nrag.gridRag(seg, numberOfLabels=int(seg.max() + 1), numberOfThreads=1)
        prev_uv_ids = rag.uvIds()
        prev_uv_ids = prev_uv_ids[(prev_uv_ids != 0).all(axis=1)]
        edge_ids = graph.findEdges(prev_uv_ids)
        assert len(edge_ids) == len(prev_uv_ids), "%i, %i" % (len(edge_ids), len(prev_uv_ids))
        assert (edge_ids != -1).all()
        prev_weights = weights[edge_ids]
        assert len(prev_uv_ids) == len(prev_weights)

        # call the agglomerator with state
        new_seg = agglomerator(affs, offsets, previous_segmentation=seg,
                               previous_edges=prev_uv_ids, previous_weights=prev_weights)

        return new_seg, seg

    print("Performing step 2...")
    from segmfriends.utils.multi_threads import ThreadPoolExecutorStackTraced
    with ThreadPoolExecutorStackTraced(n_threads) as tp:
        tasks = [tp.submit(pass2, block_id) for block_id in blocks2]
        try:
            results = [t.result() for t in tasks]
        except TypeError as e:
            print(e)


    print("Combining segmentation...")
    # Combine segmentations:
    # add offset to each of them with the maximum segm values to
    # make segmentation ids unique
    assignments = []
    for res, block_id in zip(results, blocks2):
        block = blocking.getBlockWithHalo(block_id, list(halo))
        new_seg, seeds = res
        # write out the segmentation
        inner_bb = tuple(slice(beg, end) for beg, end in zip(block.innerBlock.begin, block.innerBlock.end))
        local_bb = tuple(slice(beg, end) for beg, end in zip(block.innerBlockLocal.begin, block.innerBlockLocal.end))
        segmentation[inner_bb] = new_seg[local_bb] + id_offset

        # find the assignments to seed ids
        assignments.append(get_assignments(new_seg + id_offset, seeds))

        # Increment id_offset:
        id_offset += new_seg[local_bb].max()


    assignments = np.concatenate(assignments)

    print("Final assignment!")
    # TODO: relabel continuous to reduce max label
    # get consistent labeling with union find
    n_labels = int(segmentation.max()) + 1
    assignments = assignments[(assignments < n_labels).all(axis=1)]
    ufd = nifty.ufd.ufd(n_labels)
    ufd.merge(assignments)
    labeling = ufd.elementLabeling()

    segmentation = nifty.tools.take(labeling, segmentation)
    return segmentation
