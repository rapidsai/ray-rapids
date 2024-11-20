import ray
import cudf
import glob
import os


# create an actor pool for ease of mapping functions
from ray.util import ActorPool

@ray.remote(num_gpus=1)
class cuDFActor(object):
    def __init__(self):
        self.value = 0

    def read_parquet(self, filepath: str, columns: list = None) -> cudf.DataFrame:
        return cudf.read_parquet(filepath, columns=columns)

    def apply_udf(self, df, func, column):
        df[column] = df[column].apply(func)
        return df

    def calc_minhash(self, df, column):
        df['minhashes'] = df[column].str.minhash(width=5)
        return df

    def compute(self, filepath, func, apply_column, col_to_hash):
        df = self.read_parquet(filepath)
        df = self.apply_udf(df, func, apply_column)
        df = self.calc_minhash(df, col_to_hash)
        return df.head()


# Use all available GPUs
ray.init(dashboard_host="0.0.0.0")

# Create actors
num_gpus = 8
actors = [cuDFActor.remote() for _ in range(num_gpus)]
pool = ActorPool(actors)

in_files = glob.glob('/datasets/bzaitlen/GitRepos/RAY_PLAY/ts.parquet/*')

def f(x):
    return x + 1

results = pool.map_unordered(
    lambda actor, filepath: actor.compute.remote(
        filepath=filepath,
        func=f,
        apply_column='z',
        col_to_hash='text'
    ),
    in_files
)

results = list(results)
print(type(results[0]))
print(results[0])
print("Finished...")
