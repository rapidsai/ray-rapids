import ray
from pylibcugraph import ResourceHandle
from pylibcugraph import weakly_connected_components as pylibcugraph_wcc
from ray_comms import Comms


def _call_plc_wcc(sID, mg_graph_x, do_expensive_check):
    return pylibcugraph_wcc(
        resource_handle=ResourceHandle(Comms.get_handle(sID).getHandle()),
        graph=mg_graph_x,
        offsets=None,
        indices=None,
        weights=None,
        labels=None,
        do_expensive_check=do_expensive_check,
    )


def weakly_connected_components(input_graph, actors):
    """
    Generate the Weakly Connected Components and attach a component label to
    each vertex.

    Parameters
    ----------
    input_graph : cugraph.Graph
    actors : list of ray.actors
    """

    if input_graph.is_directed():
        raise ValueError("input graph must be undirected")

    do_expensive_check = False

    # ray.get([actor_pool[i].send_message.remote("Hello, world!", actor_pool) for i in range(pool_size)])




    # result = [
    #     client.submit(
    #         _call_plc_wcc,
    #         Comms.get_session_id(),
    #         input_graph._plc_graph[w],
    #         do_expensive_check,
    #         workers=[w],
    #         allow_other_workers=False,
    #     )
    #     for w in Comms.get_workers()
    # ]

    wait(result)

    cudf_result = [client.submit(convert_to_cudf, cp_arrays) for cp_arrays in result]

    wait(cudf_result)

    ddf = dask_cudf.from_delayed(cudf_result).persist()
    wait(ddf)
    # Wait until the inactive futures are released
    wait([(r.release(), c_r.release()) for r, c_r in zip(result, cudf_result)])

    if input_graph.renumbered:
        ddf = input_graph.unrenumber(ddf, "vertex")

    return ddf