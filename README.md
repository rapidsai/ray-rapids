# ray-rapids

The ray-rapids project is a simple suite of samples demonstrating how various [RAPIDS GPU Accelerate Data Science](https://rapids.ai/) libraries can be processed in a multi-GPU fashion with the use of [Ray Compute Engine](https://ray.io/).

The samples contained in this project are not intended to be used as is, nor this is intended to be a library, and therefore there are no installers or packages for this repository are made available.

## Getting started

To get started, a Python environment with Ray and RAPIDS libraries is necessary. There are multiple ways RAPIDS is made available, including conda, pip and Docker, please refer to the [RAPIDS Installation Guide](https://docs.rapids.ai/install/) for details. Here we will use conda for demonstration purposes.

We assume you have already conda available on your system, if not we suggest [miniforge](https://github.com/conda-forge/miniforge). You should then begin by creating a conda environment:

```
mamba create -n ray-rapids \
    -c rapidsai-nightly -c conda-forge -c nvidia \
    python=3.12 "cuda-version>=12.0,<=12.5" \
    cudf=25.02 cuml=25.02 cugraph=25.02 \
    ray-default
```

The above command assumes you will want all examples to run, including [cuDF](https://github.com/rapidsai/cudf/), [cuML](https://github.com/rapidsai/cuml/) and [cuGraph](https://github.com/rapidsai/cugraph/), but you may skip installing those that are not needed. All RAPIDS libraries require installing CUDA, conda will attempt installing the most suitable version in the range of CUDA 12.0 to CUDA 12.5, and you must make sure your system already supports CUDA 12.x. Additionally, Ray is necessary, and it is installed with the `ray-default` package.

Now with the environment already available, it must be activated:

```
conda activate ray-rapids
```

### Running

There are three samples:

- [ray-cudf](rapids-experiments/ray-cudf.py): demonstrates cuDF for minhash computing;
- [ray-kmeans](rapids-experiments/ray-kmeans.py): demonstrates cuML for k-means clustering;
- [ray-wcc](rapids-experiments/ray-wcc.py): demonstrates cuGraph for weakly connected components.

To run any of the samples, simply run them with Python:

```
cd rapids-experiments
python ray-cudf.py  # or ray-kmeas.py, or ray-wcc.py.
```

Note that since there is no installer, all files rely on relative paths, and therefore you should change into the `rapids-experiments` directory first as in the command above.
