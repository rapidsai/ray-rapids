import ray
import os
from raft_dask.common.nccl import nccl

from pylibraft.common.handle import Handle
from raft_dask.common.comms_utils import inject_comms_on_handle_coll_only
from ray_comms import Comms

class RAFTActor:
    def __init__(self, index, pool_size, session_id, actor_name_prefix="RAFT"):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(index)
        os.environ["NCCL_DEBUG"] = "DEBUG"
        self._index = index
        self._actor_name_prefix = actor_name_prefix
        self._name = f"{self._actor_name_prefix}Actor-{self._index}"
        self._pool_size = pool_size
        self._is_root = not index
        self.cb = Comms(
            verbose=True, nccl_root_location="ray-actor"
        )
        self.cb.init()
        self.unique_id = self.cb.uniqueId
        self.root_unique_id = self.unique_id if self._index == 0 else None
        self.session_id = session_id

    def broadcast_root_unique_id(self):
        if self._index == 0:
            actor_handles = [ray.get_actor(name=f"{self._actor_name_prefix}Actor-{i}", namespace=None) for i in range(1, self._pool_size)]
            futures = [actor.set_root_unique_id.remote(self.root_unique_id) for actor in actor_handles]

            # Block until all futures complete
            ray.get(futures)
        else:
            raise RuntimeError("This method should only be called by the root")

    def _setup_nccl(self):
        self._nccl = nccl()
        self._nccl.init(self._pool_size, self.root_unique_id, self._index)

    def _setup_raft(self):
        self._raft_handle = Handle(n_streams=0)

        inject_comms_on_handle_coll_only(self._raft_handle, self._nccl, self._pool_size, self._index, verbose=True)

    def _setup_post(self):
        pass

    def setup(self):
        if self.root_unique_id is None:
            raise RuntimeError(
                "The unique ID of root is not set. Make sure `broadcast_root_unique_id` "
                "runs on the root before calling this method."
            )

        try:
            print("     Setting up NCCL...", flush=True)
            self._setup_nccl()
            print("     Setting up RAFT...", flush=True)
            self._setup_raft()
            self._setup_post()
            print("     Setup complete!", flush=True)
        except Exception as e:
            print(f"An error occurred while setting up: {e}.")
            raise

    def set_root_unique_id(self, root_unique_id):
        print(f"{self._name}: set_root_unique_id")
        if self.root_unique_id is None:
            self.root_unique_id = root_unique_id
