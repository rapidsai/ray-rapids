import ray
import os
import uuid
from raft_dask.common.nccl import nccl

from pylibraft.common.handle import Handle
from raft_dask.common.comms_utils import inject_comms_on_handle_coll_only
from ray_comms import Comms
import numpy as np

from cuml_utils import make_blobs
from raft_actor import RAFTActor

@ray.remote(num_gpus=1)
class KMeansActor(RAFTActor):
    def __init__(self, index, pool_size, session_id):
        super().__init__(index=index, pool_size=pool_size, session_id=session_id, actor_name_prefix="KMeans")

    def execute(self, func, *args, **kwargs):
        """
        Dangerous!  Execute arbitrary functions
        """
        return func(self, *args, **kwargs)

    def set_new_variable(self, name, value):
        setattr(self, name, value)

    def get_variable(self, name):
        return getattr(self, name)

    def kmeans(self, n_clusters=8, max_iter=300, tol=1e-4,
                 verbose=False, random_state=1,
                 init='k-means||', n_init=1, oversampling_factor=2.0,
                 max_samples_per_batch=1<<15, convert_dtype=True,
                 output_type=None):
        from cuml.cluster.kmeans_mg import KMeansMG as cumlKMeans

        rhandle = self._raft_handle
        inp_weights = None
        inp_data = self.X
        datatype = 'cupy'

        self.kmeans_res = cumlKMeans(handle=rhandle, output_type=datatype, init=init,
                                     n_clusters=n_clusters, max_iter=max_iter, tol=tol,
                                     verbose=verbose, n_init=n_init,
                                     oversampling_factor=oversampling_factor,
                                       max_samples_per_batch=max_samples_per_batch, convert_dtype=convert_dtype).fit(
            inp_data, sample_weight=inp_weights
        )
        return True

    def score(self, vals=None, sample_weight=None):
        vals = self.X if vals is None else vals
        scores = self.kmeans_res.score(
            vals,
            sample_weight=sample_weight,
        )
        return scores

# Initialize Ray
if not ray.is_initialized():
    ray.init(include_dashboard=False, dashboard_host="0.0.0.0")

session_id = uuid.uuid4().bytes

# Start 4 Workers
pool_size = 4
actor_pool = [KMeansActor.options(name=f"KMeansActor-{i}").remote(i, pool_size, session_id=session_id) for i in range(pool_size)]
# ray.get() blocks until this completes, required before calling `setup()`
# on non-root nodes.
root_actor = ray.get_actor(name="KMeansActor-0", namespace=None)
ray.get(root_actor.broadcast_root_unique_id.remote())

# Setup Comms (NCCL/Sub-communicator)
ray.get([actor_pool[i].setup.remote() for i in range(pool_size)])

# make random blobs on each actor
make_blobs(actor_pool, int(5e3), 10, cluster_std=0.1)

# run kmeans
ray.get([actor_pool[i].kmeans.remote(n_clusters=8) for i in range(pool_size)])

scores = ray.get([actor_pool[i].score.remote() for i in range(pool_size)])

# Collect original mak_blobs data and serialized model for correctness checks
X = ray.get([actor_pool[i].get_variable.remote("X") for i in range(pool_size)])
kmeans_models = ray.get([actor_pool[i].get_variable.remote("kmeans_res") for i in range(pool_size)])

X = np.concatenate(X)
local_model = kmeans_models[0] # use any model from ray actors

expected_score = local_model.score(X)
actual_score = sum(scores)

assert abs(actual_score - expected_score) < 9e-3
print("Shutting down...", flush=True)

# Shut down Ray
ray.shutdown()
