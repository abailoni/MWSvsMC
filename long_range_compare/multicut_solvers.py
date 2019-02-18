import time
import numpy as np
import sys

# -------------------
# MULTICUT SOLVERS:
# -------------------
def solve_multicut(graph, edge_costs, p=None, solver_type="exact_solver",
                   proposal_generator_type='WS',
                   fusion_moves_kwargs=None,
                   proposal_gener_WS_kwargs=None,
                   proposal_gener_HC_kwargs=None,
                   KL_kwargs=None,
                   HC_kwargs=None):
    """
    Accepted options:

    :param solver_type: exact_solver, KL, HC, HC-KL, HC-KL-fusionMoves
    :param proposal_generator_type: WS, HC
    """


    if fusion_moves_kwargs is None:
        fusion_moves_kwargs = {'numberOfIterations': 100, # Max number of iterations
                                'stopIfNoImprovement': 10, # If no improvements, I stop earlier
                               'numberOfThreads': 1 # Parallel solutions of the fusionMove
        }
    if proposal_gener_WS_kwargs is None:
        proposal_gener_WS_kwargs = {'sigma': 2.0, # Amount of noise added
                                 'numberOfSeeds': 0.009, # Fractions of nodes that are randomly selected as seeds
                                    'seedingStrategie': "SEED_FROM_NEGATIVE"
        }
    if proposal_gener_HC_kwargs is None:
        proposal_gener_HC_kwargs = {'sigma':1.5,
                                      'weightStopCond':0.0,
                                      'nodeNumStopCond':-1.0
                                    }
    if HC_kwargs is None:
        HC_kwargs = {'weightStopCond': 0.0,  # Stop aggl. when this weight is reached
                     'nodeNumStopCond': -1.0,  # Stop aggl. when this nb. of nodes is found
                     'visitNth': 100 # How often to print
         }
    if KL_kwargs is None:
        KL_kwargs = {'numberOfInnerIterations': sys.maxsize,
                      'numberOfOuterIterations': 100,
                      'epsilon': 1e-6
         }

    # Costs to the power of p:
    if p is None or p==1:
        p = 1
        exp_costs = edge_costs.copy()
    else:
        neg_weights = edge_costs < 0.
        exp_costs = np.abs(edge_costs)**p
        exp_costs[neg_weights] *= -1

    mc_obj = graph.MulticutObjective(graph=graph, weights=exp_costs)

    tick = time.time()

    if solver_type == "exact_solver":
        log_visitor = mc_obj.loggingVisitor(verbose=True)
        solverFactory = mc_obj.multicutIlpFactory()
        solver = solverFactory.create(mc_obj)
        final_node_labels = solver.optimize(visitor=log_visitor)
    elif solver_type == "KL":
        log_visitor = mc_obj.loggingVisitor(verbose=True)
        solverFactory = mc_obj.kernighanLinFactory(**KL_kwargs)
        solver = solverFactory.create(mc_obj)
        final_node_labels = solver.optimize(visitor=log_visitor)
    elif solver_type == "HC":
        log_visitor = mc_obj.loggingVisitor(verbose=True, visitNth=100)
        solverFactory = mc_obj.greedyAdditiveFactory(**HC_kwargs)
        solver = solverFactory.create(mc_obj)
        final_node_labels = solver.optimize(visitor=log_visitor)
    elif solver_type == "HC-KL":
        log_visitor = mc_obj.loggingVisitor(verbose=False)
        solverFactory = mc_obj.greedyAdditiveFactory(**HC_kwargs)
        solver = solverFactory.create(mc_obj)
        node_labels = solver.optimize(visitor=log_visitor)
        # 2. Use a second better warm-up solver to get a better solution:
        log_visitor = mc_obj.loggingVisitor(verbose=True)
        solverFactory = mc_obj.kernighanLinFactory(**KL_kwargs)
        solver = solverFactory.create(mc_obj)
        final_node_labels = solver.optimize(visitor=log_visitor, nodeLabels=node_labels)
    elif solver_type == "HC-KL-fusionMoves":
        log_visitor = mc_obj.loggingVisitor(verbose=False)
        # 1. Initialize a warm-up solver and run optimization
        solverFactory = mc_obj.greedyAdditiveFactory(**HC_kwargs)
        solver = solverFactory.create(mc_obj)
        node_labels = solver.optimize(visitor=log_visitor)
        # 2. Use a second better warm-up solver to get a better solution:
        log_visitor = mc_obj.loggingVisitor(verbose=True)
        solverFactory = mc_obj.kernighanLinFactory(**KL_kwargs)
        solver = solverFactory.create(mc_obj)
        new_node_labels = solver.optimize(visitor=log_visitor, nodeLabels=node_labels)
        # 4. Run the funsionMuves solver
        if proposal_generator_type == "WS":
            pgen = mc_obj.watershedCcProposals(**proposal_gener_WS_kwargs)
        elif proposal_generator_type == "HC":
            pgen = mc_obj.greedyAdditiveCcProposals(**proposal_gener_HC_kwargs)
        else:
            raise ValueError("Passed type of proposal generator is not implemented")
        # fsMoveSett = mc_obj.fusionMoveSettings(mc_obj.cgcFactory(doCutPhase=True, doGlueAndCutPhase=True, mincutFactory=None,
        #     multicutFactory=None,
        #     doBetterCutPhase=False, nodeNumStopCond=0.1, sizeRegularizer=1.0))

        solverFactory = mc_obj.ccFusionMoveBasedFactory(proposalGenerator=pgen, **fusion_moves_kwargs)
        solver = solverFactory.create(mc_obj)
        final_node_labels = solver.optimize(visitor=log_visitor, nodeLabels=new_node_labels)
    else:
        raise ValueError("Passed type of solver is not implemented")

    tock = time.time()
    final_edge_labels = graph.nodesLabelsToEdgeLabels(final_node_labels)
    energy = (edge_costs * final_edge_labels).sum()

    return energy, final_node_labels, final_edge_labels, log_visitor, tock-tick
