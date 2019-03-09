from . import cremi_utils as cremi_utils

def get_experiment_by_name(name):
    assert name in globals(), "Experiment not found."
    return globals().get(name)


class CremiExperiment(object):
    def __init__(self, fixed_kwargs=None):
        if fixed_kwargs is None:
            self.fixed_kwargs = {}
        else:
            assert isinstance(fixed_kwargs, dict)
            self.fixed_kwargs = fixed_kwargs

        self.kwargs_to_be_iterated = {}

    def get_cremi_kwargs_iter(self, crop_iter, subcrop_iter,
                              init_kwargs_iter=None, nb_iterations=1):
        """
        CROPS:    Deep-z: 5     MC: 4   All: 0:4
        SUBCROPS: Deep-z: 5     MC: 6  All: 4 Tiny: 5
        """
        return cremi_utils.get_kwargs_iter(self.fixed_kwargs, self.kwargs_to_be_iterated,
                                           crop_iter=crop_iter, subcrop_iter=subcrop_iter,
                                           init_kwargs_iter=init_kwargs_iter, nb_iterations=nb_iterations)


class DebugExp(CremiExperiment):
    def __init__(self, *super_args, **super_kwargs):
        super(DebugExp, self).__init__(*super_args, **super_kwargs)

        self.fixed_kwargs.update({
            "dataset": "CREMI",
            "from_superpixels": False,
            "use_multicut": False,
            "save_segm": True,
            "WS_growing": True,
            "edge_prob": 0.0,
            "sample": "B",
            "experiment_name": "debug_exp",
            "local_attraction": False,
            "additional_model_keys": [],
            "save_UCM": False
        })

        self.kwargs_to_be_iterated.update({
            "noise_factor": [0.],
            'agglo': ["MEAN"]
        })

    def get_data(self, kwargs_iter=None):
        nb_threads_pool = 1
        nb_iterations = 1

        kwargs_iter = self.get_cremi_kwargs_iter(crop_iter=range(0, 1), subcrop_iter=range(5, 6),
                                                 init_kwargs_iter=kwargs_iter, nb_iterations=nb_iterations)

        return kwargs_iter, nb_threads_pool


class FullTestSamples(CremiExperiment):
    def __init__(self, *super_args, **super_kwargs):
        super(FullTestSamples, self).__init__(*super_args, **super_kwargs)

        self.fixed_kwargs.update({
            "dataset": "CREMI",
            "from_superpixels": True,
            "use_multicut": False,
            "save_segm": True,
            "WS_growing": False,
            "edge_prob": 0.1,
            # "sample": "B",
            "experiment_name": "FullTestSamples",
            "local_attraction": False,
            "additional_model_keys": ["debug_postproc"],
            "compute_scores": False,
            "save_UCM": False,
            "noise_factor": 0.
        })

        self.kwargs_to_be_iterated.update({
            'agglo': ["MutexWatershed", "MEAN_constr"],
            'sample': ["B+"]
            # 'sample': ["B+", "A+", "C+"]
        })

    def get_data(self, kwargs_iter=None):
        nb_threads_pool = 1
        nb_iterations = 1

        kwargs_iter = self.get_cremi_kwargs_iter(crop_iter=range(0, 1), subcrop_iter=range(6, 7),
                                                 init_kwargs_iter=kwargs_iter, nb_iterations=nb_iterations)

        return kwargs_iter, nb_threads_pool


