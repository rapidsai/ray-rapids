import cudf
import cugraph
import ray
from cugraph.dask.comms.comms_wrapper import init_subcomms as c_init_subcomms
from cugraph.dask.components.connectivity import convert_to_cudf
from cugraph.datasets import netscience
from pylibcugraph import GraphProperties, MGGraph, ResourceHandle
from pylibcugraph import weakly_connected_components as pylibcugraph_wcc
from raft_actor import RAFTActor, initialize_raft_actor_pool
from ray.util.actor_pool import ActorPool


@ray.remote(num_gpus=1)
class WCCActor(RAFTActor):
    """A Ray actor for computing weakly connected components in parallel."""

    def __init__(self, index: int, pool_size: int, session_id: bytes):
        """Initialize the WCCActor.

        Parameters
        ----------
        index: int
            The index of the actor in the pool.
        pool_size: int
            The size of the actor pool.
        session_id: bytes
            The session ID of the actor.
        """
        super().__init__(
            index=index,
            pool_size=pool_size,
            session_id=session_id,
            actor_name_prefix="WCC",
        )

    def _setup_post(self) -> None:
        """Setup the sub-communicator for cuGraph communications.

        This method is specific to cuGraph comms and is used to initialize the
        sub-communicator.
        """
        print("     Setting up cuGraph-subcom...", flush=True)
        c_init_subcomms(self._raft_handle, 2)

    def weakly_connected_components(self, start: int, stop: int) -> None:
        """Compute the weakly connected components of a graph.

        This method loads a chunk of the graph, creates a cuGraph object, and
        computes the weakly connected components using the MGGraph library.

        Parameters
        ----------
        start: int
            The start index of the chunk.
        stop: int
            The stop index of the chunk.
        """

        df = netscience.get_edgelist(download=True)
        # fake a partition with loading in a smaller subset
        df = df.iloc[start:stop]

        dg = cugraph.Graph(directed=False)

        src_array = df["src"]
        dst_array = df["dst"]
        weights = df["wgt"]

        rhandle = ResourceHandle(self._raft_handle.getHandle())

        graph_props = GraphProperties(
            is_multigraph=False,  # dg.properties.multi_edge (what is multi_edge)
            is_symmetric=not dg.graph_properties.directed,
        )
        print("Running graph creation")
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
        print(
            "Computing weakly connected components completed successfully!", flush=True
        )
        return res


def run():
    """Run the weakly connected components computation in parallel.

    This function initializes a Ray actor pool, loads the graph, computes the
    row ranges for each actor, and computes the weakly connected components in
    parallel.
    """
    # Initialize a Ray WCCActor pool
    pool_size = 4
    pool = initialize_raft_actor_pool(
        pool_size=pool_size, actor_class=WCCActor, actor_name_prefix="WCCActor"
    )
    pool = ActorPool(pool)

    # Read netscience edge list
    df = netscience.get_edgelist(download=True)

    # Compute row ranges for each actor
    row_ranges = []
    step_size = int(len(df) / pool_size)
    for i in range(pool_size):
        start = i * step_size
        stop = (i + 1) * step_size
        row_ranges.append((start, stop))

    # Compute weakly connected components on each actor
    res = list(
        pool.map(
            lambda actor, rr: actor.weakly_connected_components.remote(rr[0], rr[1]),
            row_ranges,
        )
    )

    # Combine results from all actors in single cuDF DataFrame
    wcc_ray = cudf.concat([convert_to_cudf(r) for r in res])

    # Merge computed and expected results
    df = netscience.get_graph(download=True)
    expected_dist = cugraph.weakly_connected_components(df)

    compare_dist = expected_dist.merge(
        wcc_ray, on="vertex", suffixes=["_local", "_ray"]
    )

    # Check results
    unique_local_labels = compare_dist["labels_local"].unique()
    for label in unique_local_labels.values.tolist():
        dask_labels_df = compare_dist[compare_dist["labels_local"] == label]
        dask_labels = dask_labels_df["labels_ray"]
        assert (dask_labels.iloc[0] == dask_labels).all()

    # Shut down Ray
    ray.shutdown()


if __name__ == "__main__":
    run()
