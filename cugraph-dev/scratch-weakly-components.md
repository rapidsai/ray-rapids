


CSV->edgelist mechanism
1. Read data onto each actor with cudf.read_*
  a. each actor is metaphorically like a single large dataframe partition

2. edgelist is a *special* name of a dask-cuDF dataframe:
  a. In ray this would be at least 2 columns from the source file: (source node, dest node, weight)
     from all the actors: `actor._df.edgelist`

3. Create undirected cuGraph Graph: cugraph.Graph(directed=False) and convert edgelist dataframe
  a. Can probably collapse with Step 2
  b. Create distributed version of edgelist of cuGraph Graph with SimpleDistributedGraphImpl.__from_edglist
    1. _make_plc_graph -> cythonized MGGraph pylibcugraph/graphs.pyx
    2. This is called _plc_graph but it's an MGGraph object

4. For each actor execute _call_plc_wcc on each actor's MGGraph object

# Design Options:

### Partition Representation
1. Build Function which takes NCCL handle and cuDF Dataframe and hands back a MGGraph object
2. Execute _call_plc_wcc in parallel (run function on each actor)
3. Return cuDF Dataframe or list of cupy arrays
