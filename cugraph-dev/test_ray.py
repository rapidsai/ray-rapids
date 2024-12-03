import ray
import comms
from comms import Comms
import os
from raft_dask.common.nccl import nccl


# <PID> <NCCL_UNIQUE> A
## During Init -- open port (maybe UDP) under-the-hood
# <PID> <NCCL_UNIQUE> B
## During Init -- open port (maybe UDP) under-the-hood

# In each worker setup NCCL
# Assign first worker as root ?

# @ray.remote(num_gpus=1)
# class Worker:
#    def __init__(self):
        # cb = Comms(
        #     verbose=True, nccl_root_location=root_location
        # )
        # cb.init()
   
#    def setup(self, world_size, rank):
#        collective.init_collective_group(world_size, rank, "nccl", "default")
#        return True

# import ray
# ray_context = ray.init(dashboard_port="0.0.0.0")

# # imperative
# num_workers = 2
# workers = []
# init_rets = []
# for i in range(num_workers):
#    w = Worker.remote()
#    workers.append(w)
#    init_rets.append(w.setup.remote(num_workers, i))
# results = ray.get([w.compute.remote() for w in workers])

@ray.remote(num_gpus=1)
class MyActor:
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
        # print(f"{type(self.root_uniqueId)=}")

    # _func_init_nccl
    def setup_nccl(self):
        try:
            n = nccl()
            print("     Begin NCCL init", flush=True)
            n.init(self._pool_size, self.root_uniqueId, self._index)
            print("     End NCCL init", flush=True)
        except Exception as e:
            print(f"An error occurred initializing NCCL: {e}.")
            raise  
    
    def set_root_uniqueID(self, root_uniqueId):
        if self.root_uniqueId is None:
            self.root_uniqueId = root_uniqueId

    def get_root_uniqueID(self):
        return self.root_uniqueId

    def send_message(self, message, actor_pool):

        for i in range(self._pool_size):
            # if i == self._index and i != 0:
            #     print(f"get unique id from actor {i=}", flush=True)
            #     self.root_uniqueID = ray.get(actor_pool[0].get_root_uniqueID.remote())
            #     print(f"{self.root_uniqueID=}, {len(actor_pool)}")
            #     actor_pool[i].set_root_uniqueID.remote(self.root_uniqueID)

            if i != self._index:
                print("Send to Actor", i, flush=True)
                actor_pool[i].receive_message.remote(message, self._index)
                
        print("Starting NCCL", flush=True)
        self.setup_nccl()
        print("NCCL is setup", flush=True)


    def receive_message(self, message, sender_index):
        print(f"Actor {self._index} received message from Actor {sender_index}: {message}", flush=True)

# Initialize Ray
ray.init(dashboard_host=None)

pool_size = 3
actor_pool = [MyActor.remote(i, pool_size) for i in range(pool_size)]
parent_unique = ray.get(actor_pool[0].get_root_uniqueID.remote())

# set root uniqueId
[a.set_root_uniqueID.remote(parent_unique) for a in actor_pool]


# Send a message from Actor 0 to all other actors in the pool
# ray.get([actor_pool[0].send_message.remote("Hello, world!", actor_pool)])
ray.get([actor_pool[i].send_message.remote("Hello, world!", actor_pool) for i in range(pool_size)])

# Keep the program running to see the output
import time
time.sleep(1)

# Shut down Ray
ray.shutdown()