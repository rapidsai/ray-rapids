import cudf
import lipsum.generate_sentences

cdf = cudf.datasets.timeseries(dtypes={"name": str, "id": int, "x": float, "y": float})
cdf = cdf.reset_index()
cdf['z'] = 1
df = cdf.to_pandas()
df['text'] = df.apply(lambda x: generate_lorem_ipsum(), axis=1)

ddf = dd.from_pandas(df, npartitions=16)
ddf.to_parquet('ts.parquet')
