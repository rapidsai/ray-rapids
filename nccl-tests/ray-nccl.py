import ray
import os
from raft_dask.common.nccl import nccl

from ray_comms import Comms


@ray.remote(num_gpus=1)
class NCCLActor:
    def __init__(self, index, pool_size):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(index)
        os.environ["NCCL_DEBUG"] = "DEBUG"
        self._index = index
        self._pool_size = pool_size
        self._is_root = True if not index else False
        self.cb = Comms(
            verbose=True, nccl_root_location='ray-actor'
        )
        self.cb.init()
        self.uniqueId = self.cb.uniqueId
        self.root_uniqueId = self.uniqueId if self._index == 0 else None
    
    # _func_init_nccl
    def setup_nccl(self):
        try:
            n = nccl()
            print("     Begin NCCL init", flush=True)
            n.init(self._pool_size, self.root_uniqueId, self._index)
            print("     End NCCL init", flush=True)
            return True
        except Exception as e:
            print(f"An error occurred initializing NCCL: {e}.")
            raise  
            return False
    
    def set_root_uniqueID(self, root_uniqueId):
        if self.root_uniqueId is None:
            self.root_uniqueId = root_uniqueId

    def get_root_uniqueID(self):
        return self.root_uniqueId

    def send_message(self, message, actor_pool):
        for i in range(self._pool_size):
            if i != self._index:
                print("Send to Actor", i, flush=True)
                actor_pool[i].receive_message.remote(message, self._index)
                

    def receive_message(self, message, sender_index):
        print(f"Actor {self._index} received message from Actor {sender_index}: {message}", flush=True)

# Initialize Ray
if not ray.is_initialized():
    ray.init(include_dashboard=False)

# Start 3 Workers
pool_size = 3
actor_pool = [NCCLActor.remote(i, pool_size) for i in range(pool_size)]

# Get the uniqueID from rank-0 and broadcast to all other workers
parent_unique = ray.get(actor_pool[0].get_root_uniqueID.remote())
[a.set_root_uniqueID.remote(parent_unique) for a in actor_pool]

ray.get([actor_pool[i].setup_nccl.remote() for i in range(pool_size)])


# ray.get([actor_pool[i].send_message.remote("Hello, world!", actor_pool) for i in range(pool_size)])


# # Shut down Ray
# ray.shutdown()