import numpy as np
import os
import json
from segmfriends.utils import yaml2dict, parse_data_slice, check_dir_and_create

from signet.cluster import Cluster
from sklearn.metrics import adjusted_rand_score
import signet.block_models as signetMdl
from scipy import sparse
from nifty.tools import fromAdjMatrixToEdgeList, fromEdgeListToAdjMatrix
from nifty.graph import UndirectedGraph

from segmfriends.utils.config_utils import adapt_configs_to_model, recursive_dict_update, return_recursive_key_in_dict

from copy import deepcopy

from GASP.segmentation import run_GASP
import time

from segmfriends.algorithms.multicut.multicut import multicut

import matplotlib
import matplotlib.pyplot as plt


def from_adj_matrix_to_edge_list(sparse_adj_matrix):
    # sparse_adj_matrix.setdiag(np.zeros(sparse_adj_matrix.shape[0], sparse_adj_matrix.dtype))
    nb_edges = sparse_adj_matrix.count_nonzero()
    if not isinstance(sparse_adj_matrix, np.ndarray):
        sparse_adj_matrix = sparse_adj_matrix.toarray()
    # sh = sparse_adj_matrix.shape[0]
    # nb_edges = int((sh*sh - sh) / 2)
    edge_list = np.empty((nb_edges, 3))

    # Set diagonal elements to zero, we don't care about them:
    real_nb_edges = fromAdjMatrixToEdgeList(sparse_adj_matrix, edge_list, 1)
    edge_list = edge_list[:real_nb_edges]
    uvIds = edge_list[:,:2].astype('uint64')
    edge_weights = edge_list[:,2].astype('float32')
    return uvIds, edge_weights


def from_edge_list_to_adj_matrix(uvIds, edge_weights):
    edge_list = np.concatenate((uvIds, np.expand_dims(edge_weights, axis=-1)),
                               axis=1)
    nb_nodes = int(edge_list[:,:2].max()+1)
    A_p = np.zeros((nb_nodes, nb_nodes))
    A_n = np.zeros((nb_nodes, nb_nodes))
    fromEdgeListToAdjMatrix(A_p, A_n, edge_list, 1)
    return A_p, A_n


def run_method_on_graph(method_type, true_assign,
                        k=None, p=None, n=None,
                        spectral_method_name=None,
                        linkage_criteria=None,
                        add_cannot_link_constraints=None,
                        signed_edge_weights=None,
                        multicut_solver_type="kernighanLin",
                        graph=None, A_p=None, A_n=None,
                        experiment_name=None,
                        eta=None,
                        project_directory=None,
                        save_output_segm=False,
                        output_shape=None):
    # Run clustering:
    tick = time.time()
    # print(method_type)
    if method_type == "GASP":
        # TODO: assert
        print(linkage_criteria, add_cannot_link_constraints)
        node_labels, _ = run_GASP(graph, signed_edge_weights,
                                        # edge_sizes=edge_sizes,
                                        linkage_criteria=linkage_criteria,
                                        add_cannot_link_constraints=add_cannot_link_constraints,
                                        use_efficient_implementations=False,
                                        # **additional_kwargs
                                        )
    elif method_type == "multicut":
        print(multicut_solver_type)
        node_labels = multicut(graph, None, None, signed_edge_weights, solver_type=multicut_solver_type)
    elif method_type == "spectral":
        c = Cluster((A_p, A_n))
        print(spectral_method_name)
        try:
            if spectral_method_name == "BNC":
                node_labels = c.spectral_cluster_bnc(k=k, normalisation='sym')
            elif spectral_method_name == "L-sym":
                node_labels = c.spectral_cluster_laplacian(k = k, normalisation='sym')
            elif spectral_method_name == "SPONGE":
                # FIXME: not sure about this...
                # node_labels = c.geproblem_laplacian(k = k, normalisation='additive')
                node_labels = c.SPONGE(k = k)
            elif spectral_method_name == "SPONGE-sym":
                # node_labels = c.geproblem_laplacian(k = k, normalisation='multiplicative')
                node_labels = c.SPONGE_sym(k=k)
            else:
                raise NotImplementedError
        except np.linalg.LinAlgError:
            print("#### LinAlgError ({}) ####".format(spectral_method_name))
            return
    else:
        raise NotImplementedError

    runtime = time.time() - tick

    # Compute scores and stats:
    RAND_score = adjusted_rand_score(node_labels, true_assign)
    counts = np.bincount(node_labels.astype('int64'))
    nb_clusters = (counts > 0).sum()
    biggest_clusters  = np.sort(counts)[::-1][:10]

    # Save stuff/results...
    print(runtime, RAND_score)
    # print(nb_clusters, biggest_clusters)

    # Save config setup:
    new_results = {}

    # TODO: delete this crap
    new_results["method_type"] = method_type
    new_results["k"] = k
    new_results["p"] = p
    new_results["n"] = n
    new_results["eta"] = eta
    new_results["spectral_method_name"] = spectral_method_name
    new_results["linkage_criteria"] = linkage_criteria
    new_results["add_cannot_link_constraints"] = add_cannot_link_constraints
    new_results["multicut_solver_type"] = multicut_solver_type
    new_results["experiment_name"] = experiment_name
    new_results["project_directory"] = project_directory
    ID = str(np.random.randint(1000000000))
    new_results["ID"] = ID

    experiment_dir_path = os.path.join(project_directory, experiment_name)
    check_dir_and_create(experiment_dir_path)
    check_dir_and_create(os.path.join(experiment_dir_path, 'scores'))
    result_file = os.path.join(experiment_dir_path, 'scores',
                               '{}_{}_{}_{}.json'.format(ID, method_type, spectral_method_name, linkage_criteria))

    # Actually save results:
    new_results["runtime"] = runtime
    new_results["RAND_score"] = RAND_score
    new_results["nb_clusters"] = int(nb_clusters)
    new_results["biggest_clusters"] = [int(size) for size in biggest_clusters]


    with open(result_file, 'w') as f:
        json.dump(new_results, f, indent=4, sort_keys=True)


    # Save output:
    if save_output_segm:
        assert output_shape is not None
        check_dir_and_create(os.path.join(experiment_dir_path, 'out_segms'))
        export_file = os.path.join(experiment_dir_path, 'out_segms',
                                   '{}_{}_{}.h5'.format(method_type, spectral_method_name, linkage_criteria))
        from segmfriends.utils.various import writeHDF5
        writeHDF5(node_labels.reshape(output_shape), export_file, 'segm', compression='gzip')


def get_kwargs_iter(fixed_kwargs, kwargs_to_be_iterated,
                    init_kwargs_iter=None, nb_iterations=1):
    kwargs_iter = init_kwargs_iter if isinstance(init_kwargs_iter, list) else []

    iter_collected = {
    }

    KEYS_TO_ITER = ['method_type', 'eta', 'spectral_method_name', 'linkage_criteria', 'add_cannot_link_constraints', 'p']
    for key in KEYS_TO_ITER:
        if key in fixed_kwargs:
            iter_collected[key] = [fixed_kwargs[key]]
        elif key in kwargs_to_be_iterated:
            iter_collected[key] = kwargs_to_be_iterated[key]
        else:
            raise ValueError("Iter key {} was not passed!".format(key))

    fixed_kwargs = deepcopy(fixed_kwargs)
    dataset = fixed_kwargs.pop("dataset")

    # Load the data:
    for _ in range(nb_iterations):
        if dataset == "SSBM":
            n = fixed_kwargs.get("n")
            # p = fixed_kwargs.pop("p")
            k = fixed_kwargs.get("k")
            for p in iter_collected['p']:
                for eta in iter_collected['eta']:
                    print("Creating SSBM model...")
                    (A_p, A_n), true_assign = signetMdl.SSBM(n=n, k=k, pin=p, etain=eta, values='gaussian')

                    A_signed = A_p - A_n
                    uv_ids, signed_edge_weights = from_adj_matrix_to_edge_list(A_signed)

                    print("Building nifty graph...")
                    graph = UndirectedGraph(n)
                    graph.insertEdges(uv_ids)
                    nb_edges = graph.numberOfEdges
                    assert graph.numberOfEdges == uv_ids.shape[0]

                    # Test connected components:
                    from nifty.graph import components
                    components = components(graph)
                    components.build()
                    print("Nb. connected components in graph:", np.unique(components.componentLabels()).shape)

                    # Symmetrize matrices:
                    grid = np.indices((n, n))
                    matrix_mask = grid[0] > grid[1]
                    A_p = matrix_mask * A_p.toarray()
                    A_n = matrix_mask * A_n.toarray()
                    A_p = A_p + np.transpose(A_p)
                    A_n = A_n + np.transpose(A_n)
                    A_p = sparse.csr_matrix(A_p)
                    A_n = sparse.csr_matrix(A_n)

                    # Start collecting kwargs:
                    for method_type in iter_collected["method_type"]:
                        if method_type == "spectral":
                            for spectral_method_name in iter_collected["spectral_method_name"]:
                                new_kwargs = {}
                                new_kwargs.update(fixed_kwargs)

                                iterated_kwargs = {
                                    'method_type': method_type,
                                    'true_assign': true_assign,
                                    'spectral_method_name': spectral_method_name,
                                    'A_p': A_p,
                                    'A_n': A_n,
                                    'n': n,
                                    'eta': eta,
                                    'p': p,
                                    'k': k,
                                }
                                new_kwargs.update({k: v for k, v in iterated_kwargs.items() if k not in new_kwargs})
                                kwargs_iter.append(new_kwargs)
                        elif method_type == "GASP":
                            for linkage in iter_collected["linkage_criteria"]:
                                for CNC in iter_collected["add_cannot_link_constraints"]:
                                    new_kwargs = {}
                                    new_kwargs.update(fixed_kwargs)

                                    iterated_kwargs = {
                                        'method_type': method_type,
                                        'true_assign': true_assign,
                                        'linkage_criteria': linkage,
                                        'add_cannot_link_constraints': CNC,
                                        'signed_edge_weights': signed_edge_weights,
                                        'graph': graph,
                                        'eta': eta,
                                        'n': n,
                                        'p': p,
                                        'k': k,

                                    }
                                    new_kwargs.update({k: v for k, v in iterated_kwargs.items() if k not in new_kwargs})
                                    kwargs_iter.append(new_kwargs)
                        elif method_type == "multicut":
                            new_kwargs = {}
                            new_kwargs.update(fixed_kwargs)

                            iterated_kwargs = {
                                'method_type': method_type,
                                'true_assign': true_assign,
                                'signed_edge_weights': signed_edge_weights,
                                'graph': graph,
                                'eta': eta,
                                'n': n,
                                'p': p,
                                'k': k,

                            }
                            new_kwargs.update({k: v for k, v in iterated_kwargs.items() if k not in new_kwargs})
                            kwargs_iter.append(new_kwargs)
                        else:
                            raise NotImplementedError
        else:
            raise NotImplementedError

    return kwargs_iter

def get_kwargs_iter_CREMI(fixed_kwargs, kwargs_to_be_iterated,
                    init_kwargs_iter=None, nb_iterations=1):
    kwargs_iter = init_kwargs_iter if isinstance(init_kwargs_iter, list) else []

    iter_collected = {
    }

    KEYS_TO_ITER = ['method_type', 'spectral_method_name', 'linkage_criteria', 'add_cannot_link_constraints']
    for key in KEYS_TO_ITER:
        if key in fixed_kwargs:
            iter_collected[key] = [fixed_kwargs[key]]
        elif key in kwargs_to_be_iterated:
            iter_collected[key] = kwargs_to_be_iterated[key]
        else:
            raise ValueError("Iter key {} was not passed!".format(key))

    fixed_kwargs = deepcopy(fixed_kwargs)
    dataset = fixed_kwargs.pop("dataset")

    # Load the data:
    for _ in range(nb_iterations):
        if dataset == "CREMI":
            from .load_datasets import get_dataset_data, CREMI_crop_slices, get_dataset_offsets
            affs, GT = get_dataset_data(dataset='CREMI', sample="B", crop_slice_str=":,18:28,1020:1120,990:1090", run_connected_components=True)


            n = 10000
            # p = fixed_kwargs.pop("p")
            print("Creating pixel graph:")
            from GASP.utils.graph import build_pixel_lifted_graph_from_offsets
            offsets = np.array(get_dataset_offsets("CREMI"))
            graph, is_local_edge, true_assign, edge_sizes = build_pixel_lifted_graph_from_offsets(GT.shape, offsets, GT_label_image=GT)
            true_assign = true_assign.astype('uint64')
            assert (edge_sizes==1).all()
            uv_ids = graph.uvIds()

            k = np.unique(true_assign).shape[0]

            # TODO: check sign of affinities
            edge_weights = graph.edgeValues(np.rollaxis(affs, 0, 4))

            # FIXME: always use additive cost for the moment...
            # Compute log costs:
            signed_edge_weights = edge_weights - 0.5

            print("Create positive and negative graph adj. matrices:")

            A_p, A_n = from_edge_list_to_adj_matrix(uv_ids, signed_edge_weights)
            A_p = sparse.csr_matrix(A_p)
            A_n = sparse.csr_matrix(A_n)

            # Dump data:
            experiment_dir_path = os.path.join(fixed_kwargs['project_directory'], fixed_kwargs['experiment_name'])
            check_dir_and_create(experiment_dir_path)
            check_dir_and_create(os.path.join(experiment_dir_path, 'segms'))
            check_dir_and_create(os.path.join(experiment_dir_path, 'out_segms'))
            export_file = os.path.join(experiment_dir_path, 'out_segms',
                                       'inputs.h5')
            from segmfriends.utils.various import writeHDF5
            writeHDF5(affs.astype('float32'), export_file, 'affs', compression='gzip')
            writeHDF5(GT, export_file, 'gt', compression='gzip')

            # Start collecting kwargs:
            for method_type in iter_collected["method_type"]:
                if method_type == "spectral":
                    for spectral_method_name in iter_collected["spectral_method_name"]:
                        new_kwargs = {}
                        new_kwargs.update(fixed_kwargs)

                        iterated_kwargs = {
                            'method_type': method_type,
                            'true_assign': true_assign,
                            'spectral_method_name': spectral_method_name,
                            'A_p': A_p,
                            'A_n': A_n,
                            'n': n,
                            'eta': None,
                            'p': None,
                            'k': k,
                            'save_output_segm': True,
                            'output_shape': GT.shape
                        }
                        new_kwargs.update({k: v for k, v in iterated_kwargs.items() if k not in new_kwargs})
                        kwargs_iter.append(new_kwargs)
                elif method_type == "GASP":
                    for linkage in iter_collected["linkage_criteria"]:
                        for CNC in iter_collected["add_cannot_link_constraints"]:
                            new_kwargs = {}
                            new_kwargs.update(fixed_kwargs)

                            iterated_kwargs = {
                                'method_type': method_type,
                                'true_assign': true_assign,
                                'linkage_criteria': linkage,
                                'add_cannot_link_constraints': CNC,
                                'signed_edge_weights': signed_edge_weights,
                                'graph': graph,
                                'eta': None,
                                'n': n,
                                'p': None,
                                'k': k,
                                'save_output_segm': True,
                                'output_shape': GT.shape

                            }
                            new_kwargs.update({k: v for k, v in iterated_kwargs.items() if k not in new_kwargs})
                            kwargs_iter.append(new_kwargs)
                elif method_type == "multicut":
                    new_kwargs = {}
                    new_kwargs.update(fixed_kwargs)

                    iterated_kwargs = {
                        'method_type': method_type,
                        'true_assign': true_assign,
                        'signed_edge_weights': signed_edge_weights,
                        'graph': graph,
                        'eta': None,
                        'n': n,
                        'p': None,
                        'k': k,
                        'save_output_segm': True,
                        'output_shape': GT.shape

                    }
                    new_kwargs.update({k: v for k, v in iterated_kwargs.items() if k not in new_kwargs})
                    kwargs_iter.append(new_kwargs)
                else:
                    raise NotImplementedError
        else:
            raise NotImplementedError

    return kwargs_iter





def get_experiment_by_name(name):
    assert name in globals(), "Experiment not found."
    return globals().get(name)



class SSBMExperiment(object):
    def __init__(self, fixed_kwargs=None):
        if fixed_kwargs is None:
            self.fixed_kwargs = {}
        else:
            assert isinstance(fixed_kwargs, dict)
            self.fixed_kwargs = fixed_kwargs

        self.kwargs_to_be_iterated = {}

        self.fixed_kwargs.update({
            "n": 10000, # 10000
            "k": 100, # 100
            # "eta": 0.35,
            "dataset": "SSBM",
            "experiment_name": "first_experiment",
        })

        self.kwargs_to_be_iterated.update({
            'spectral_method_name': ["SPONGE", "BNC", "L-sym", "SPONGE-sym"],
            'linkage_criteria': ["mean", "abs_max", "sum"],
            # 'linkage_criteria': ["mean"],
            "eta": np.linspace(0.01, 0.4, num=40),
            "method_type": ["GASP", "spectral"],
            # "method_type": ["GASP"],
            'add_cannot_link_constraints': [False],
            'p': [0.1],
        })

    def get_data(self, kwargs_iter=None, nb_threads_pool = 1):
        nb_iterations = 1

        kwargs_iter = get_kwargs_iter(self.fixed_kwargs, self.kwargs_to_be_iterated, init_kwargs_iter=kwargs_iter, nb_iterations=nb_iterations)

        return kwargs_iter, nb_threads_pool


    def get_plot_data(self, path, sort_by='eta'):
        """
        :param sort_by: 'noise_factor' or 'long_range_prob'
        """
        # TODO: use experiment path

        # Create dictionary:
        results_collected = {}

        for item in os.listdir(path):
            if os.path.isfile(os.path.join(path, item)):
                filename = item
                if not filename.endswith(".json") or filename.startswith("."):
                    continue
                # ID, sample, agglo_type, _ = filename.split("_")
                result_file = os.path.join(path, filename)
                with open(result_file, 'rb') as f:
                    file_dict = json.load(f)

                if sort_by == 'eta':
                    sort_key = file_dict["eta"]
                else:
                    raise ValueError
                method = file_dict["method_type"]
                if method == "GASP":
                    method_descriptor = file_dict["linkage_criteria"] + str(file_dict["add_cannot_link_constraints"])
                elif method == "spectral":
                    method_descriptor = file_dict["spectral_method_name"]
                elif method == "multicut":
                    method_descriptor = file_dict["multicut_solver_type"]
                else:
                    raise ValueError
                new_results = {}
                new_results[method] = {}
                new_results[method][method_descriptor] = {}
                new_results[method][method_descriptor][sort_key] = {}
                new_results[method][method_descriptor][sort_key][file_dict["ID"]] = file_dict

                try:
                    results_collected = recursive_dict_update(new_results,results_collected)
                except KeyError:
                    continue
        return results_collected


    def make_plots(self, project_directory):
        # self.collect_scores(project_directory)


        exp_name = self.fixed_kwargs['experiment_name']
        scores_path = os.path.join(project_directory, exp_name, "scores")
        results_collected = self.get_plot_data(scores_path)

        key_x = ['eta']
        key_y = ['RAND_score']
        key_value = ['runtime']

        legend_axes = {
            'eta': "Eta",
            'RAND_score': "RAND",
            'runtime': "runtime",
        }

        # Find best values for every crop:
        matplotlib.rcParams.update({'font.size': 9})
        ncols, nrows = 1, 1
        f, all_ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(7, 7))
        ax = all_ax

        label_names = []

        type_counter = 0
        for method in [ty for ty in ['GASP', 'spectral', 'multicut'] if ty in results_collected]:
            for method_descriptor in results_collected[method]:
                label_names.append(method_descriptor)
                # color_list.append(colors[agglo_type][non_link][local_attraction])
                sub_dict = results_collected[method][method_descriptor]
                values = []
                etas = []

                for eta in sub_dict:
                    multiple_values = []
                    for ID in sub_dict[eta]:
                        data_dict = sub_dict[eta][ID]
                        multiple_values.append(return_recursive_key_in_dict(data_dict, key_value))
                    if len(multiple_values) == 0:
                        continue
                    multiple_values = np.array(multiple_values)
                    mean = multiple_values.mean(axis=0)
                    values.append(mean)
                    etas.append(eta)

                # Sort keys:
                etas = np.array(etas)
                values = np.array(values)
                argsort = np.argsort(etas)

                ax.plot(etas[argsort], values[argsort], label=label_names[-1])
                type_counter += 0

        ax.set_xlabel(legend_axes[key_x[-1]])
        ax.set_ylabel(legend_axes[key_y[-1]])
        lgnd = ax.legend()

        # ax.set_yscale("log", nonposy='clip')

        # for i in range(10):
        #     try:
        #         lgnd.legendHandles[i]._sizes = [30]
        #     except IndexError:
        #         break

        # ax.set_title("CREMI training sample {}".format(sample))

        # if sample == "B":
        #     ax.set_ylim([0.080, 0.090])
        # else:
        ax.autoscale(enable=True, axis='both')

        # if all_keys[-1] == 'runtime':
        #     ax.set_yscale("log", nonposy='clip')

        # plt.subplots_adjust(left=0.05, right=0.95, wspace=0.3)

        plot_dir = os.path.join(project_directory, exp_name, "plots")
        check_dir_and_create(plot_dir)

        # f.suptitle("Crop of CREMI sample {} (90 x 300 x 300)".format(sample))
        f.savefig(os.path.join(plot_dir,
                               'method_comparison.pdf'),
                  format='pdf')



class CremiExperiment(SSBMExperiment):
    def __init__(self, fixed_kwargs=None):
        # super(CremiExperiment, self).__init__(fixed_kwargs)
        if fixed_kwargs is None:
            self.fixed_kwargs = {}
        else:
            assert isinstance(fixed_kwargs, dict)
            self.fixed_kwargs = fixed_kwargs

        self.kwargs_to_be_iterated = {}


        self.kwargs_to_be_iterated = {}

        self.fixed_kwargs.update({
            "n": 10000, # 10000
            # "k": 100, # 100
            # "eta": 0.35,
            "dataset": "CREMI",
            "experiment_name": "first_cremi_experiment",
        })


        self.kwargs_to_be_iterated.update({
            'spectral_method_name': ["SPONGE", "BNC", "L-sym", "SPONGE-sym"],
            'linkage_criteria': ["mean", "abs_max", "sum"],
            # 'linkage_criteria': ["mean"],
            # "eta": np.linspace(0.01, 0.4, num=40),
            "method_type": ["GASP", "spectral"],
            # "method_type": ["GASP"],
            'add_cannot_link_constraints': [False],
            'p': [0.1],
        })

    def get_data(self, kwargs_iter=None, nb_threads_pool = 1):
        nb_iterations = 1

        kwargs_iter = get_kwargs_iter_CREMI(self.fixed_kwargs, self.kwargs_to_be_iterated, init_kwargs_iter=kwargs_iter, nb_iterations=nb_iterations)

        return kwargs_iter, nb_threads_pool
