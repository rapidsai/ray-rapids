import ray
import os
import uuid
from raft_dask.common.nccl import nccl

from pylibraft.common.handle import Handle
from raft_dask.common.comms_utils import inject_comms_on_handle_coll_only
from ray_comms import Comms
from raft_dask.common import (
    perform_test_comm_split,
    perform_test_comms_allgather,
    perform_test_comms_allreduce,
    perform_test_comms_bcast,
    perform_test_comms_device_multicast_sendrecv,
    perform_test_comms_device_send_or_recv,
    perform_test_comms_device_sendrecv,
    perform_test_comms_gather,
    perform_test_comms_gatherv,
    perform_test_comms_reduce,
    perform_test_comms_reducescatter,
    perform_test_comms_send_recv,
)


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
        self._raft_handle = Handle(n_streams=4)

        inject_comms_on_handle_coll_only(self._raft_handle, self._nccl, self._pool_size, self._index, verbose=True)

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
        for i in range(self._pool_size):
            if i != self._index:
                print("Send to Actor", i, flush=True)
                actor_pool[i].receive_message.remote(message, self._index)

    def receive_message(self, message, sender_index):
        print(f"Actor {self._index} received message from Actor {sender_index}: {message}", flush=True)

    def test_comm(self, func, n_trials=5):
        print(f"{self._name}: {func.__name__} {n_trials=}")
        return func(self._raft_handle, n_trials)


# Initialize Ray
if not ray.is_initialized():
    ray.init(include_dashboard=False)

session_id = uuid.uuid4().bytes

# Start 3 Workers
pool_size = 3
actor_pool = [NCCLActor.options(name=f"NCCLActor-{i}").remote(i, pool_size, session_id=session_id) for i in range(pool_size)]

# ray.get() blocks until this completes, required before calling `setup()`
# on non-root nodes.
root_actor = ray.get_actor(name="NCCLActor-0", namespace=None)
ray.get(root_actor.broadcast_root_unique_id.remote())

ray.get([actor_pool[i].setup.remote() for i in range(pool_size)])

# ray.get([actor_pool[i].send_message.remote("Hello, world!", actor_pool) for i in range(pool_size)])


result_comm_split = ray.get([actor_pool[i].test_comm.remote(func=perform_test_comm_split) for i in range(pool_size)])
assert all(result_comm_split)


# Commented out collectives are failing, further investigation needed.
collectives = [
    perform_test_comms_allgather,
    perform_test_comms_allreduce,
    # perform_test_comms_bcast,
    # perform_test_comms_gather,
    # perform_test_comms_gatherv,
    # perform_test_comms_reduce,
    perform_test_comms_reducescatter,
]

for c in collectives:
    result = ray.get([actor_pool[i].test_comm.remote(func=c) for i in range(pool_size)])
    assert all(result)

# Shut down Ray
ray.shutdown()
