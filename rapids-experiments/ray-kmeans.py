from typing import Any

import numpy as np
import ray
from raft_actor import RAFTActor, initialize_raft_actor_pool

from cuml_utils import make_blobs


@ray.remote(num_gpus=1)
class KMeansActor(RAFTActor):
    """A Ray actor class for performing K-Means clustering.

    Parameters
    ----------
    index: int
        Index of the actor in the pool.
    pool_size: int
        Size of the actor pool.
    session_id: bytes
        Session ID for the actor.

    Attributes
    ----------
    X: Any
        Input data for K-Means clustering.
    kmeans_res: Any
        Result of K-Means clustering.
    """

    def __init__(self, index: int, pool_size: int, session_id: bytes):
        """Initialize the KMeansActor.

        Parameters
        ----------
        index: int
            Index of the actor in the pool.
        pool_size: int
            Size of the actor pool.
        session_id: bytes
            Session ID for the actor.
        """
        super().__init__(
            index=index,
            pool_size=pool_size,
            session_id=session_id,
            actor_name_prefix="KMeans",
        )

    def execute(self, func, *args, **kwargs):
        """Execute an arbitrary function on the actor.

        Warning: This method is dangerous and should be used with caution.

        Parameters
        ----------
        func: callable
            Function to be executed.
        *args: Any
            Variable arguments for the function.
        **kwargs: Any
            Keyword arguments for the function.

        Returns
        -------
        ret: Any
            Result of the function execution.
        """
        return func(self, *args, **kwargs)

    def set_new_variable(self, name: str, value: Any) -> None:
        """Set a new variable on the actor.

        Parameters
        ----------
        name: str
            Name of the variable.
        value: Any
            Value of the variable.
        """
        setattr(self, name, value)

    def get_variable(self, name: str) -> Any:
        """Get a variable from the actor.

        Parameters
        ----------
        name: str
            Name of the variable.

        Returns
        -------
        ret: Any
            Value of the variable.
        """
        return getattr(self, name)

    def kmeans(
        self,
        n_clusters=8,
        max_iter=300,
        tol=1e-4,
        verbose=False,
        random_state=1,
        init="k-means||",
        n_init=1,
        oversampling_factor=2.0,
        max_samples_per_batch=1 << 15,
        convert_dtype=True,
        output_type=None,
    ):
        """Perform K-Means clustering on the input data.

        Parameters
        ----------
        n_clusters: int, optional
            Number of clusters (default=8).
        max_iter: int, optional
            Maximum number of iterations (default=300).
        tol: float, optional
            Tolerance for convergence (default=1e-4).
        verbose: bool, optional
            Verbose output (default=False).
        random_state: int, optional
            Random seed (default=1).
        init: str, optional
            Initialization method (default="k-means||").
        n_init: int, optional
            Number of initializations (default=1).
        oversampling_factor: float, optional
            Oversampling factor (default=2.0).
        max_samples_per_batch: int, optional
            Maximum number of samples per batch (default=1 << 15).
        convert_dtype: bool, optional
            Convert data type (default=True).
        output_type: Any, optional
            Output type (default=None).
        """
        from cuml.cluster.kmeans_mg import KMeansMG as cumlKMeans

        rhandle = self._raft_handle
        inp_weights = None
        inp_data = self.X
        datatype = "cupy"

        self.kmeans_res = cumlKMeans(
            handle=rhandle,
            output_type=datatype,
            init=init,
            n_clusters=n_clusters,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
            n_init=n_init,
            oversampling_factor=oversampling_factor,
            max_samples_per_batch=max_samples_per_batch,
            convert_dtype=convert_dtype,
        ).fit(inp_data, sample_weight=inp_weights)

    def score(self, vals=None, sample_weight=None):
        """Compute the score of the K-Means model.

        Parameters
        ----------
        vals: Any, optional
            Input data (default=None).
        sample_weight: Any, optional
            Sample weights (default=None).

        Returns
        -------
        ret: Any
            Score of the K-Means model.
        """
        vals = self.X if vals is None else vals
        scores = self.kmeans_res.score(
            vals,
            sample_weight=sample_weight,
        )
        return scores


def run():
    """Run the k-means clustering example.

    This function initializes a KMeansActor pool, generates random blobs on each actor,
    runs k-means clustering, and computes the score of the model.
    """
    # Initialize KMeansActor Pool
    pool_size = 4
    pool = initialize_raft_actor_pool(
        pool_size=pool_size, actor_class=KMeansActor, actor_name_prefix="KMeansActor"
    )

    # make random blobs on each actor
    make_blobs(pool, int(5e3), 10, cluster_std=0.1)

    # run kmeans
    ray.get([pool[i].kmeans.remote(n_clusters=8) for i in range(pool_size)])

    scores = ray.get([pool[i].score.remote() for i in range(pool_size)])

    # Collect original make_blobs data and serialized model for correctness checks
    X = ray.get([pool[i].get_variable.remote("X") for i in range(pool_size)])
    kmeans_models = ray.get(
        [pool[i].get_variable.remote("kmeans_res") for i in range(pool_size)]
    )

    X = np.concatenate(X)
    local_model = kmeans_models[0]  # use any model from ray actors

    expected_score = local_model.score(X)
    actual_score = sum(scores)

    assert abs(actual_score - expected_score) < 9e-3
    print("Shutting down...", flush=True)

    # Shut down Ray
    ray.shutdown()


if __name__ == "__main__":
    run()
