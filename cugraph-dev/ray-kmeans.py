import ray
import os
import uuid
from raft_dask.common.nccl import nccl

from pylibraft.common.handle import Handle
from raft_dask.common.comms_utils import inject_comms_on_handle_coll_only
from ray_comms import Comms
import cudf
from ray.util.actor_pool import ActorPool

from cuml_utils import make_blobs

@ray.remote(num_gpus=1)
class NCCLActor:
    def __init__(self, index, pool_size, session_id):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(index)
        os.environ["NCCL_DEBUG"] = "DEBUG"
        self._index = index
        self._name = f"NCCLActor-{self._index}"
        self._pool_size = pool_size
        self._is_root = True if not index else False
        self.cb = Comms(
            verbose=True, nccl_root_location='ray-actor'
        )
        self.cb.init()
        self.uniqueId = self.cb.uniqueId
        self.root_uniqueId = self.uniqueId if self._index == 0 else None
        self.session_id = session_id

    def broadcast_root_unique_id(self):
        if self._index == 0:
            actor_handles = [ray.get_actor(name=f"NCCLActor-{i}", namespace=None) for i in range(1, pool_size)]
            futures = [actor.set_root_unique_id.remote(self.root_uniqueId) for actor in actor_handles]

            # Block until all futures complete
            ray.get(futures)

            return True
        else:
            raise RuntimeError("This method should only be called by the root")

    def _setup_nccl(self):
        self._nccl = nccl()
        self._nccl.init(self._pool_size, self.root_uniqueId, self._index)

    def _setup_raft(self):
        self._raft_handle = Handle(n_streams=0)

        inject_comms_on_handle_coll_only(self._raft_handle, self._nccl, self._pool_size, self._index, verbose=True)

    def execute(self, func, *args, **kwargs):
        """
        Dangerous!  Execute arbitrary functions
        """
        return func(self, *args, **kwargs)

    def setup(self):
        if self.root_uniqueId is None:
            raise RuntimeError(
                "The unique ID of root is not set. Make sure `broadcast_root_unique_id` "
                "runs on the root before calling this method."
            )

        try:
            print("     Setting up NCCL...", flush=True)
            self._setup_nccl()
            print("     Setting up RAFT...", flush=True)
            self._setup_raft()
            print("     Setup complete!", flush=True)
        except Exception as e:
            print(f"An error occurred while setting up: {e}.")
            raise

    def set_root_unique_id(self, root_uniqueId):
        print(f"{self._name}: set_root_unique_id")
        if self.root_uniqueId is None:
            self.root_uniqueId = root_uniqueId

    def send_message(self, message, actor_pool):
        # for i in range(self._pool_size):
            # if i != self._index:
                # actor_pool[i].receive_message.remote(message, self._index)
        futures = [actor_pool[i].receive_message.remote(message, self._index) for i in range(len(actor_pool)) if i != self._index]

    def receive_message(self, message, sender_index):
        print(f"Actor {self._index} received message from Actor {sender_index}: {message}", flush=True)

    def set_new_variable(self, name, value):
        setattr(self, name, value)

    def get_variable(self, name):
        return getattr(self, name)

    def kmeans(self):
        """
        sample
        weight
        """
        from cuml.cluster.kmeans_mg import KMeansMG as cumlKMeans

        rhandle = self._raft_handle
        # if not has_weights:
        #     inp_data = concatenate(objs)
        #     inp_weights = None
        # else:
        #     inp_data = concatenate([X for X, weights in objs])
        #     inp_weights = concatenate([weights for X, weights in objs])
        inp_weights = None
        inp_data = self.X
        datatype = 'cupy'

        self.kmeans_res = cumlKMeans(handle=rhandle, output_type=datatype).fit(
            inp_data, sample_weight=inp_weights
        )

        print("succeded", flush=True)
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
    # ray.init(include_dashboard=False, address="10.33.227.163:6379")
    ray.init(include_dashboard=False, dashboard_host="0.0.0.0", _temp_dir="/datasets/bzaitlen/RAY_TEMP")

session_id = uuid.uuid4().bytes

# Start 4 Workers
pool_size = 4
actor_pool = [NCCLActor.options(name=f"NCCLActor-{i}").remote(i, pool_size, session_id=session_id) for i in range(pool_size)]
import time
time.sleep(2)
# ray.get() blocks until this completes, required before calling `setup()`
# on non-root nodes.
root_actor = ray.get_actor(name="NCCLActor-0", namespace=None)
ray.get(root_actor.broadcast_root_unique_id.remote())

# Setup Comms (NCCL/Sub-communicator)
ray.get([actor_pool[i].setup.remote() for i in range(pool_size)])

# res = ray.get([actor_pool[i].send_message.remote("Hello, world!", actor_pool) for i in range(pool_size)])
ray.get([actor_pool[i].send_message.remote("Hello, world!", actor_pool) for i in range(pool_size)])

make_blobs(actor_pool, 1001, 10, centers=42, cluster_std=0.1)

res = ray.get([actor_pool[i].kmeans.remote() for i in range(pool_size)])

# res = ray.get([actor_pool[i].get_variable.remote("X") for i in range(pool_size)])
# print(res, flush=True)

# pool = ActorPool(actor_pool)

# ray.get(actor_pool[i].kmeans.remote(...)
scores = ray.get([actor_pool[i].score.remote() for i in range(pool_size)])

import time
# time.sleep(10000)
print("Shutting down...", flush=True)

# Shut down Ray
# ray.shutdown()