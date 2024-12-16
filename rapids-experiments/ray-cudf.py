import os

import cudf
import numpy as np
import ray
import rmm
from ray.util import ActorPool

# This is necessary to work around a pynvjitlink bug.
os.environ["CUDA_CACHE_DISABLE"] = "1"


@ray.remote(num_gpus=1)
class cuDFActor:
    """A remote actor class for performing cuDF operations.

    This class utilizes the RAPIDS cuDF library to perform various data manipulation
    tasks on a GPU. It is designed to be used with the Ray distributed computing
    framework.
    """

    def __init__(self):
        """Initialize the cuDFActor.

        Sets up the CUDA memory resource and initializes the internal DataFrame.
        """
        # TODO: Check if there's an initial pool
        mr = rmm.mr.CudaAsyncMemoryResource()
        rmm.mr.set_current_device_resource(mr)

        self.df = None

    def generate_data(self, start_end: tuple) -> cudf.DataFrame:
        """Generate a sample timeseries dataset.

        Parameters
        ----------
        start_end: tuple
            A tuple containing the start and end dates of the timeseries.

        Returns
        -------
        df: cudf.DataFrame
            A cuDF DataFrame containing the generated timeseries data.
        """
        start, end = start_end
        df = cudf.datasets.timeseries(
            start=start,
            end=end,
            freq="1d",
            dtypes={"name": str, "id": int, "x": float, "y": float},
        )
        return df

    def filter(self, df: cudf.DataFrame, expr: str) -> cudf.DataFrame:
        """Filter a cuDF DataFrame based on a query expression.

        Parameters
        ----------
        df: cudf.DataFrame
            The cuDF DataFrame to be filtered.
        expr: str
            The query expression to filter the DataFrame.

        Returns
        -------
        df: cudf.DataFrame
            The filtered cuDF DataFrame.
        """
        return df.query(expr)

    def apply_udf(
        self, df: cudf.DataFrame, func: callable, column: str
    ) -> cudf.DataFrame:
        """Apply a user-defined function to a column of a cuDF DataFrame.

        Parameters
        ----------
        df: cudf.DataFrame
            The cuDF DataFrame to be modified.
        func: callable
            The user-defined function to be applied.
        column: str
            The name of the column to apply the function to.

        Returns
        -------
        df: cudf.DataFrame
            The modified cuDF DataFrame.
        """
        df[column] = df[column].apply(func)
        return df

    def calc_minhash(self, df: cudf.DataFrame, column: str) -> cudf.DataFrame:
        """Calculate the minhash values for a column of a cuDF DataFrame.

        Parameters
        ----------
        df: cudf.DataFrame
            The cuDF DataFrame to be modified.
        column: str
            The name of the column to calculate minhash values for.

        Returns
        -------
        df: cudf.DataFrame
            The modified cuDF DataFrame with minhash values.
        """
        a = cudf.Series([1], dtype=np.uint32)
        b = cudf.Series([2], dtype=np.uint32)
        df["minhashes"] = df[column].str.minhash(0, a=a, b=b, width=5)
        return df

    def compute(
        self,
        start_end: tuple,
        expr: str,
        func: callable,
        apply_column: str,
        col_to_hash: str,
    ) -> cudf.DataFrame:
        """Perform a series of operations on a cuDF DataFrame.

        This method generates a sample timeseries dataset, filters it based on a
        query expression, applies a user-defined function to a column, calculates
        minhash values for another column, and returns the resulting DataFrame.

        Parameters
        ----------
        start_end: tuple
            A tuple containing the start and end dates of the timeseries.
        expr: str
            The query expression to filter the DataFrame.
        func: callable
            The user-defined function to be applied.
        apply_column: str
            The name of the column to apply the function to.
        col_to_hash: str
            The name of the column to calculate minhash values for.

        Returns
        -------
        df: cudf.DataFrame
            The resulting cuDF DataFrame.
        """
        df = self.generate_data(start_end)  # or read_parquet
        df = self.filter(df, expr)
        df = self.apply_udf(df, func, apply_column)
        df = self.calc_minhash(df, col_to_hash)
        return df.head()


def run():
    """Runs a parallel computation using Ray and cuDF actors.

    This function initializes a Ray cluster, creates a pool of cuDF actors, and
    uses the pool to compute a user-defined function on simulated data in parallel.
    """
    # Create actors
    num_actors = 4
    actors = [cuDFActor.remote() for _ in range(num_actors)]
    pool = ActorPool(actors)

    # define udf
    def f(x):
        return x + 1

    # start / end dates for simulated data
    start_end = [(f"24-{i:02d}-01", f"24-{i:02d}-20") for i in range(1, num_actors + 1)]

    results = pool.map_unordered(
        lambda actor, start_end: actor.compute.remote(
            start_end=start_end,
            expr="x > 0",
            func=f,
            apply_column="x",
            col_to_hash="name",
        ),
        start_end,
    )

    results = list(results)
    print(results[0].head())
    print("Finished...")
    ray.shutdown()


if __name__ == "__main__":
    run()
