import ray
import os
import uuid
from raft_dask.common.nccl import nccl

from pylibraft.common.handle import Handle
from raft_dask.common.comms_utils import inject_comms_on_handle_coll_only
from ray_comms import Comms
import cudf
from ray.util.actor_pool import ActorPool
import cugraph
from simpleDistributedGraph import simpleDistributedGraphImpl
from pylibcugraph import MGGraph, ResourceHandle, GraphProperties



@ray.remote(num_gpus=1)
class NCCLActor:
    def __init__(self, index, pool_size, session_id):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(index)
        os.environ["NCCL_DEBUG"] = "TRACE"
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


    def _setup_subcom(self):
        from cugraph.dask.comms.comms_wrapper import init_subcomms as c_init_subcomms
        import math


        def __get_2D_div(ngpus):
            prows = int(math.sqrt(ngpus))
            while ngpus % prows != 0:
                prows = prows - 1
            return prows, int(ngpus / prows)


        def subcomm_init(prows=2, pcols=1):
            ngpus = self._pool_size
            if prows is None and pcols is None:
                if partition_type == 1:
                    pcols, prows = __get_2D_div(ngpus)
                else:
                    prows, pcols = __get_2D_div(ngpus)
            else:
                if prows is not None and pcols is not None:
                    if ngpus != prows * pcols:
                        raise Exception(
                            "prows*pcols should be equal to the\
        number of processes"
                        )
                elif prows is not None:
                    if ngpus % prows != 0:
                        raise Exception(
                            "prows must be a factor of the number\
        of processes"
                        )
                    pcols = int(ngpus / prows)
                elif pcols is not None:
                    if ngpus % pcols != 0:
                        raise Exception(
                            "pcols must be a factor of the number\
        of processes"
                        )
                    prows = int(ngpus / pcols)

            self.__subcomm = (prows, pcols)


        # subcomm_init(prows, pcols)
        c_init_subcomms(self._raft_handle, 2)
        

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
            print("     Setting up cuGraph-subcom...", flush=True)
            self._setup_subcom()
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

    def load_csv(self, start, stop):
        """
        1. Each actor loads in a chunk
        2. Each actor has a NCCL/Raft Handle
        3. Pass each chunk and handle to MGGraph
        """
        
        df = cudf.read_csv('/home/nfs/bzaitlen/.cugraph/datasets/netscience.csv', 
                                nrows=stop-start, 
                                skiprows=start, 
                                header=None, 
                                sep=' ', 
                                names=["src", "dst", "wgt"])
        # edge = cugraph.from_cudf_edgelist(df, "src", "dst", "wgt")
        dg = cugraph.Graph(directed=False)

        # from_dask_cudf_edgelist(dg, ddf, "src", "dst", "wgt")

        src_array = df['src'] 
        dst_array = df['dst']
        weights = df['wgt']
        # print(dir(self._raft_handle), flush=True)
        rhandle = ResourceHandle(self._raft_handle.getHandle())
        # TypeError: Argument 'graph_properties' has incorrect type (expected pylibcugraph.graph_properties.GraphProperties, got Properties)
        print("len: ", len(src_array), flush=True)
        graph_props = GraphProperties(
            is_multigraph=False, #dg.properties.multi_edge (what is multi_edge)
            is_symmetric=True, #not dg.graph_properties.directed,
        )
        plc_graph = MGGraph(
            resource_handle=rhandle,
            graph_properties=graph_props,
            src_array=[src_array],
            dst_array=[dst_array],
            weight_array=[weights],
            edge_id_array=None,
            edge_type_array=None,
            num_arrays=1,
            store_transposed=False,
            symmetrize=True,
            do_expensive_check=False,
            drop_multi_edges=True,
        )
        return df.tail()

# Initialize Ray
if not ray.is_initialized():
    ray.init(include_dashboard=False)

session_id = uuid.uuid4().bytes

# Start 3 Workers
pool_size = 2
actor_pool = [NCCLActor.options(name=f"NCCLActor-{i}").remote(i, pool_size, session_id=session_id) for i in range(pool_size)]

# ray.get() blocks until this completes, required before calling `setup()`
# on non-root nodes.
root_actor = ray.get_actor(name="NCCLActor-0", namespace=None)
ray.get(root_actor.broadcast_root_unique_id.remote())

ray.get([actor_pool[i].setup.remote() for i in range(pool_size)])

row_ranges = [(0, 1800), (1800, 3600), (3600, 5484)]
pool = ActorPool(actor_pool)
print(list(pool.map_unordered(lambda actor, rr: actor.load_csv.remote(rr[0], rr[1]),
                              row_ranges)))


# Shut down Ray
ray.shutdown()