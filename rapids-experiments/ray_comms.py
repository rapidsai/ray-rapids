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
import uuid

from raft_dask.common.nccl import nccl

logger = logging.getLogger(__name__)


class Comms:
    """
    Initializes and manages underlying NCCL comms handles across the a pool of
    Ray actors. It is expected that `init()` will be called explicitly. It is
    recommended to also call `destroy()` when the comms are no longer needed so
    the underlying resources can be cleaned up. This class is not meant to be
    thread-safe.
    """

    valid_nccl_placements = "ray-actor"

    def __init__(
        self,
        verbose: bool = False,
        nccl_root_location: str = "ray-actor",
    ) -> None:
        """
        Construct a new CommsContext instance

        Parameters
        ----------
        verbose : bool
                  Print verbose logging
        nccl_root_location : string
                  Indicates where the NCCL's root node should be located.
                  ['client', 'worker', 'scheduler' (default), 'ray-actor']

        """
        self.nccl_root_location = nccl_root_location.lower()
        if self.nccl_root_location not in Comms.valid_nccl_placements:
            raise ValueError(
                f"nccl_root_location must be one of: " f"{Comms.valid_nccl_placements}"
            )

        self.sessionId = uuid.uuid4().bytes

        self.nccl_initialized = False

        self.verbose = verbose

        if verbose:
            print("Initializing comms!")

    def __del__(self) -> None:
        if self.nccl_initialized:
            self.destroy()

    def create_nccl_uniqueid(self) -> None:
        self.uniqueId = nccl.get_unique_id()

    def init(self) -> None:
        """
        Initializes the underlying comms.
        """
        self.create_nccl_uniqueid()

        self.nccl_initialized = True

        if self.verbose:
            print("Initialization complete.")
