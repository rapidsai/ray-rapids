# Copyright (c) 2020-2024, NVIDIA CORPORATION.
#
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
#

import logging
import math
import os
import time
import uuid
import warnings
from collections import OrderedDict

from dask.distributed import default_client
from dask_cuda.utils import nvml_device_index

from pylibraft.common.handle import Handle

from raft_dask.common.comms_utils import (
    inject_comms_on_handle,
    inject_comms_on_handle_coll_only,
)
from raft_dask.common.nccl import nccl

logger = logging.getLogger(__name__)


class Comms:
    """
    Initializes and manages underlying NCCL and UCX comms handles across
    the workers of a Dask cluster. It is expected that `init()` will be
    called explicitly. It is recommended to also call `destroy()` when
    the comms are no longer needed so the underlying resources can be
    cleaned up. This class is not meant to be thread-safe.

    Examples
    --------
    .. code-block:: python

        # The following code block assumes we have wrapped a C++
        # function in a Python function called `run_algorithm`,
        # which takes a `raft::handle_t` as a single argument.
        # Once the `Comms` instance is successfully initialized,
        # the underlying `raft::handle_t` will contain an instance
        # of `raft::comms::comms_t`

        from dask_cuda import LocalCUDACluster
        from dask.distributed import Client

        from raft.dask.common import Comms, local_handle

        cluster = LocalCUDACluster()
        client = Client(cluster)

        def _use_comms(sessionId):
            return run_algorithm(local_handle(sessionId))

        comms = Comms(client=client)
        comms.init()

        futures = [client.submit(_use_comms,
                                 comms.sessionId,
                                 workers=[w],
                                 pure=False) # Don't memoize
                       for w in cb.worker_addresses]
        wait(dfs, timeout=5)

        comms.destroy()
        client.close()
        cluster.close()
    """

    valid_nccl_placements = ("ray-actor")

    def __init__(
        self,
        comms_p2p=False,
        verbose=False,
        streams_per_handle=0,
        nccl_root_location="ray-actor",
    ):
        """
        Construct a new CommsContext instance

        Parameters
        ----------
        comms_p2p : bool
                    Initialize UCX endpoints?
        verbose : bool
                  Print verbose logging
        nccl_root_location : string
                  Indicates where the NCCL's root node should be located.
                  ['client', 'worker', 'scheduler' (default), 'ray-actor']

        """
        self.comms_p2p = comms_p2p

        self.nccl_root_location = nccl_root_location.lower()
        if self.nccl_root_location not in Comms.valid_nccl_placements:
            raise ValueError(
                f"nccl_root_location must be one of: "
                f"{Comms.valid_nccl_placements}"
            )

        self.streams_per_handle = streams_per_handle

        self.sessionId = uuid.uuid4().bytes

        self.nccl_initialized = False
        self.ucx_initialized = False

        self.verbose = verbose

        if verbose:
            print("Initializing comms!")

    def __del__(self):
        if self.nccl_initialized or self.ucx_initialized:
            self.destroy()

    def create_nccl_uniqueid(self):
        self.uniqueId = nccl.get_unique_id()
        
    def init(self, workers=None):
        """
        Initializes the underlying comms. NCCL is required but
        UCX is only initialized if `comms_p2p == True`

        Parameters
        ----------
        workers : Sequence
                  Unique collection of workers for initializing comms.
        """
        self.create_nccl_uniqueid()
        
        self.nccl_initialized = True

        if self.comms_p2p:
            self.ucx_initialized = True

        if self.verbose:
            print("Initialization complete.")

def __get_2D_div(ngpus):
    prows = int(math.sqrt(ngpus))
    while ngpus % prows != 0:
        prows = prows - 1
    return prows, int(ngpus / prows)


def subcomm_init(prows=2, pcols=2, ngpus=1):
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

    return (prows, pcols)