# Copyright (c) 2020-2024, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc

import pytest

import cugraph
import cugraph.dask as dcg
from cugraph.datasets import netscience

from simpleDistributedGraph import simpleDistributedGraphImpl


# =============================================================================
# Pytest Setup / Teardown - called for each test function
# =============================================================================


def setup_function():
    gc.collect()


# =============================================================================
# Parameters
# =============================================================================


DATASETS = [netscience]
# Directed graph is not currently supported
IS_DIRECTED = [False, True]


# =============================================================================
# Helper
# =============================================================================

def get_n_workers():
    from dask.distributed import default_client

    client = default_client()
    return len(client.scheduler_info()["workers"])


def get_chunksize(input_path):
    """
    Calculate the appropriate chunksize for dask_cudf.read_csv
    to get a number of partitions equal to the number of GPUs.

    Examples
    --------
    >>> import cugraph.dask as dcg
    >>> chunksize = dcg.get_chunksize(datasets_path / 'netscience.csv')

    """

    import os
    from glob import glob
    import math

    input_files = sorted(glob(str(input_path)))
    if len(input_files) == 1:
        size = os.path.getsize(input_files[0])
        chunksize = math.ceil(size / get_n_workers())
    else:
        size = [os.path.getsize(_file) for _file in input_files]
        chunksize = max(size)
    return chunksize


def get_dask_edgelist(self, download=False):
    """
    Return a distributed Edgelist.

    Parameters
    ----------
    download : Boolean (default=False)
        Automatically download the dataset from the 'url' location within
        the YAML file.
    """
    if self._edgelist is None:
        full_path = self.get_path()
        if not full_path.is_file():
            if download:
                full_path = self.__download_csv(self.metadata["url"])
            else:
                raise RuntimeError(
                    f"The datafile {full_path} does not"
                    " exist. Try setting download=True"
                    " to download the datafile"
                )

        header = None
        if isinstance(self.metadata["header"], int):
            header = self.metadata["header"]

        blocksize = get_chunksize(full_path)
        print(f"{blocksize=}")
        self._edgelist = dask_cudf.read_csv(
            path=full_path,
            blocksize=blocksize,
            delimiter=self.metadata["delim"],
            names=self.metadata["col_names"],
            dtype={
                self.metadata["col_names"][i]: self.metadata["col_types"][i]
                for i in range(len(self.metadata["col_types"]))
            },
            header=header,
        )

    return self._edgelist.copy()


def get_ray_edgelist(dataset, download=False):
    import dask_cudf
    print("get_ray_edgelist", flush=True)
    # breakpoint()
    # if dataset._edgelist is None:
    if True:
        full_path = dataset.get_path()
        if not full_path.is_file():
            if download:
                full_path = dataset.__download_csv(dataset.metadata["url"])
            else:
                raise RuntimeError(
                    f"The datafile {full_path} does not"
                    " exist. Try setting download=True"
                    " to download the datafile"
                )

        header = None
        if isinstance(dataset.metadata["header"], int):
            header = dataset.metadata["header"]

        blocksize = dcg.get_chunksize(full_path)
        print(f"{blocksize=}", flush=True)
        dataset._edgelist = dask_cudf.read_csv(
            path=full_path,
            blocksize=blocksize,
            delimiter=dataset.metadata["delim"],
            names=dataset.metadata["col_names"],
            dtype={
                dataset.metadata["col_names"][i]: dataset.metadata["col_types"][i]
                for i in range(len(dataset.metadata["col_types"]))
            },
            header=header,
        )

    return dataset._edgelist.copy()

def from_dask_cudf_edgelist(
    self,
    input_ddf,
    source="source",
    destination="destination",
    edge_attr=None,
    weight=None,
    edge_id=None,
    edge_type=None,
    renumber=True,
    store_transposed=False,
):
    """
    Initializes the distributed graph from the dask_cudf.DataFrame
    edgelist. By default, renumbering is enabled to map the source and destination
    vertices into an index in the range [0, V) where V is the number
    of vertices.  If the input vertices are a single column of integers
    in the range [0, V), renumbering can be disabled and the original
    external vertex ids will be used.
    Note that the graph object will store a reference to the
    dask_cudf.DataFrame provided.

    Parameters
    ----------
    input_ddf : dask_cudf.DataFrame
        The edgelist as a dask_cudf.DataFrame

    source : str or array-like, optional (default='source')
        Source column name or array of column names

    destination : str, optional (default='destination')
        Destination column name or array of column names

    edge_attr : str or List[str], optional (default=None)
        Names of the edge attributes.  Can either be a single string
        representing the weight column name, or a list of length 3
        holding [weight, edge_id, edge_type].  If this argument is
        provided, then the weight/edge_id/edge_type arguments must
        be left empty.

    weight : str, optional (default=None)
        Name of the weight column in the input dataframe.

    edge_id : str, optional (default=None)
        Name of the edge id column in the input dataframe.

    edge_type : str, optional (default=None)
        Name of the edge type column in the input dataframe.

    renumber : bool, optional (default=True)
        If source and destination indices are not in range 0 to V where V
        is number of vertices, renumber argument should be True.

    store_transposed : bool, optional (default=False)
        If True, stores the transpose of the adjacency matrix.  Required
        for certain algorithms.

    """

    if self._Impl is None:
        self._Impl = simpleDistributedGraphImpl(self.graph_properties)
    elif type(self._Impl) is not simpleDistributedGraphImpl:
        raise RuntimeError("Graph is already initialized")
    elif self._Impl.edgelist is not None:
        raise RuntimeError("Graph already has values")
    self._Impl._simpleDistributedGraphImpl__from_edgelist(
        input_ddf,
        source=source,
        destination=destination,
        edge_attr=edge_attr,
        weight=weight,
        edge_id=edge_id,
        edge_type=edge_type,
        renumber=renumber,
        store_transposed=store_transposed,
    )


def get_mg_graph(dataset, directed):
    """Returns an MG graph"""
    print(f"{dataset._edgelist=}", flush=True)
    # ddf = dataset.get_dask_edgelist()
    ddf = get_ray_edgelist(dataset)
    print(f"{type(ddf)=}", flush=True)
    print(f"{ddf=}", flush=True)

    dg = cugraph.Graph(directed=directed)
    # breakpoint()
    # dg.from_dask_cudf_edgelist(ddf, "src", "dst", "wgt")
    from_dask_cudf_edgelist(dg, ddf, "src", "dst", "wgt")

    return dg


# =============================================================================
# Tests
# =============================================================================


# @pytest.mark.mg
# @pytest.mark.parametrize("dataset", DATASETS)
# @pytest.mark.parametrize("directed", IS_DIRECTED)
def test_dask_mg_wcc(dataset, directed):
    input_data_path = dataset.get_path()
    print(f"dataset={input_data_path}")

    g = dataset.get_graph(create_using=cugraph.Graph(directed=directed))
    print(g._plc_graph)
    dg = get_mg_graph(dataset, directed)
    print(dg._plc_graph)

    # breakpoint()
    if not directed:
        expected_dist = cugraph.weakly_connected_components(g)
        result_dist = dcg.weakly_connected_components(dg)
        # result_dist =  ray_cugraph.weakly_connected_components(dg)

        result_dist = result_dist.compute()
        compare_dist = expected_dist.merge(
            result_dist, on="vertex", suffixes=["_local", "_dask"]
        )

        unique_local_labels = compare_dist["labels_local"].unique()

        for label in unique_local_labels.values.tolist():
            dask_labels_df = compare_dist[compare_dist["labels_local"] == label]
            dask_labels = dask_labels_df["labels_dask"]
            assert (dask_labels.iloc[0] == dask_labels).all()
    else:
        with pytest.raises(ValueError):
            cugraph.weakly_connected_components(g)

    print("SUCCESS", flush=True)

if __name__ == "__main__":
    from dask_cuda import LocalCUDACluster
    from dask.distributed import Client
    import cugraph.dask.comms.comms as Comms

    cluster = LocalCUDACluster(n_workers=2)
    dask_client = Client(cluster)
    Comms.initialize(p2p=False)

    test_dask_mg_wcc(netscience, False)
    test_dask_mg_wcc(netscience, True) #pytest raises


    # cluster clean up
    Comms.destroy()
    dask_client.close()
    cluster.close()
