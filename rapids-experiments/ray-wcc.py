import ray
import os
import uuid
from raft_dask.common.nccl import nccl

from pylibraft.common.handle import Handle
from raft_dask.common.comms_utils import inject_comms_on_handle_coll_only
from raft_actor import RAFTActor
from ray_comms import Comms
import cudf
from ray.util.actor_pool import ActorPool
import cugraph
from pylibcugraph import MGGraph, ResourceHandle, GraphProperties
from pylibcugraph import weakly_connected_components as pylibcugraph_wcc
from cugraph.datasets import netscience
from cugraph.dask.components.connectivity import convert_to_cudf

from cugraph.dask.comms.comms_wrapper import init_subcomms as c_init_subcomms


@ray.remote(num_gpus=1)
class WCCActor(RAFTActor):
    def __init__(self, index, pool_size, session_id):
        super().__init__(index=index, pool_size=pool_size, session_id=session_id, actor_name_prefix="WCC")

    def _setup_subcom(self):
        # setup sub-communicator (specific to cuGraph comms)
        # TODO: Understand partitioning of prows / pcols from subcomm_init
        c_init_subcomms(self._raft_handle, 2)

    def _setup_post(self):
        print("     Setting up cuGraph-subcom...", flush=True)
        self._setup_subcom()

    def weakly_connected_components(self, start, stop):
        """
        1. Each actor loads in a chunk
        2. Each actor has a NCCL/Raft Handle
        3. Pass each chunk and handle to MGGraph
        """

        df = netscience.get_edgelist(download=True)
        # fake a partition with loading in a smaller subset
        df = df.iloc[start:stop]

        dg = cugraph.Graph(directed=False)

        src_array = df['src']
        dst_array = df['dst']
        weights = df['wgt']

        rhandle = ResourceHandle(self._raft_handle.getHandle())

        graph_props = GraphProperties(
            is_multigraph=False, #dg.properties.multi_edge (what is multi_edge)
            is_symmetric=not dg.graph_properties.directed,
        )
        print("running graph creation")
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
            symmetrize=False,
            do_expensive_check=False,
            drop_multi_edges=True,
        )
        res = pylibcugraph_wcc(
                resource_handle=rhandle,
                graph=plc_graph,
                offsets=None,
                indices=None,
                weights=None,
                labels=None,
                do_expensive_check=False,
            )
        print("succeded", flush=True)
        return res

# Initialize Ray
if not ray.is_initialized():
    ray.init(include_dashboard=False)

session_id = uuid.uuid4().bytes

# Start 4 Workers
pool_size = 4
actor_pool = [WCCActor.options(name=f"WCCActor-{i}").remote(i, pool_size, session_id=session_id) for i in range(pool_size)]

# ray.get() blocks until this completes, required before calling `setup()`
# on non-root nodes.
root_actor = ray.get_actor(name="WCCActor-0", namespace=None)
ray.get(root_actor.broadcast_root_unique_id.remote())

# Setup Comms (NCCL/Sub-communicator)
ray.get([actor_pool[i].setup.remote() for i in range(pool_size)])
df = netscience.get_edgelist(download=True)

row_ranges = []
step_size = int(len(df) / pool_size)
for i in range(pool_size):
    start = i * step_size
    stop = (i + 1) * step_size
    row_ranges.append((start, stop))

pool = ActorPool(actor_pool)

res = list(pool.map(lambda actor, rr: actor.weakly_connected_components.remote(rr[0], rr[1]),
                              row_ranges))

wcc_ray = cudf.concat([convert_to_cudf(r) for r in res])

df = netscience.get_graph(download=True)
expected_dist = cugraph.weakly_connected_components(df)

compare_dist = expected_dist.merge(
    wcc_ray, on="vertex", suffixes=["_local", "_ray"]
)

unique_local_labels = compare_dist["labels_local"].unique()

for label in unique_local_labels.values.tolist():
    dask_labels_df = compare_dist[compare_dist["labels_local"] == label]
    dask_labels = dask_labels_df["labels_ray"]
    assert (dask_labels.iloc[0] == dask_labels).all()

# Shut down Ray
ray.shutdown()
