# atracks

[![License BSD-3](https://img.shields.io/github/license/uvarc/atracks?label=license&style=flat)](https://github.com/uvarc/atracks/blob/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/atracks.svg?color=green)](https://pypi.org/project/atracks)
[![Python Version](https://img.shields.io/pypi/pyversions/atracks.svg?color=green)](https://python.org)
[![codecov](https://codecov.io/gh/uvarc/atracks/branch/main/graph/badge.svg)](https://codecov.io/gh/uvarc/atracks)

Plugin to track spindle poles in mitotic cells over time. It leverages [Cellpose](https://www.cellpose.org/), [OpenCV](https://github.com/opencv/opencv-python), and [Scikit-Image](https://scikit-image.org/) for segmentation and [Prefect](https://www.prefect.io/) and [Dask](https://www.dask.org/) for workflow orchestration.

## Installation 

You can install `atracks` via [pip]:

    pip install atracks


To install latest development version:

    pip install git+https://github.com/uvarc/atracks.git

## Image file formats

MitosisAnalyzer is reading image files using the [Bioio](https://github.com/bioio-devs/bioio) package. A variety of plugins exist to support common image file formats, including .tiff, .ome-tiff, .zarr, .nd2, .czi, .lif, etc.. By installing these additional bioio plugins you can easily expand MitosisAnalyzer's ability to process a large variety of image formats without the need to touch the source code.  

## Running the MitosisAnalyzer application

In a command line shell, run the following command:
```
atracks -i imagestack.nd2 -o my_outputdir -s 1 -d 2 -r 1
```

**Command line arguments:**

```
-h, --help                              show this help message and exit
-i INPUT, --input INPUT                 single image file or directory with image files to be processed
-o OUTPUT, --output OUTPUT              output file or directory
-s SPINDLE, --spindle SPINDLE           channel # for tracking spindle poles
-d DNA, --dna DNA                       channel # for tracking dna
-r REFFRAME, --refframe REFFRAME        reference frame to determine spindle pole axis (0=autodetect based on cell long axis)
-t THRESHOLD, --threshold THRESHOLD     threshold of cytoplasmic background signal in spindle channel; value relative to max 
                                            spindle intensity 0.0-1.0 (0.0=autodetect using Otsu)
-b BLUR, --blur BLUR                    applies a gaussian blur before segmenting spindle poles. The value determines the 
                                            blurring radius; a value of 0 omits blurring.
-c, --cellpose, --no-cellpose           use Cellpose to detect cell contour
-f FRAMERATE, --framerate FRAMERATE     number of frames per second
-e EXECUTOR, --executor EXECUTOR        set executor. Options: sequential, concurrent, dask
--address ADDRESS                       provide address to existing Dask Scheduler. Default: local, spins up a new Dask 
                                            scheduler and clients on localhost using the --process and --threads options.
--processes PROCESSES                   number of parallel processes. Ignored when --adress is set.
--threads THREADS                       number of threads per process. Ignored when --address is set.
```                  

## Advanced Settings

### Prefect

Processing of images is orchestrated as [Prefect](https://www.prefect.io/) workflows. You may set up a Prefect API key which enables you to authenticate a local environment to work with [Prefect Cloud](https://docs.prefect.io/v3/manage/cloud/index). **Prefect Cloud is an optional tool to help you observe execution of your analysis pipeline; the MitosisAnalyzer executes just fine without Prefect Cloud.**

### Dask

By default, MitosisAnalyzer is executing pipeline tasks on a single computer/node in local concurrent mode. For large processing workloads that involve very large or many image files, you may choose to distribute tasks via [Dask](https://www.dask.org/) using the `-e/--executor` flag, see above. By default, a local Dask cluster is spun up with the specified number of processes, each process with a single thread. The `--processes` and `--threads` options override these defaults and set the number of workers (processess) and threads per worker process. The local Dask Cluster is torn down after the analysis run is finished. 

Alternatively, you may connect execution of the MitosisAnalyzer pipeline to an existing Dask cluster, using the `--address` command line argument, see above. The existing Dask cluster may run locally or on a different host, assuming that host can be reached from your execution environment.

When using the `-e dask` option, images are read as delayed Dask arrays, meaning that IO operations occur only for the image chunks needed at the specific computational step. This a key aspect contributing to the pipeline's scalability. 

**Additonal environment variables:**

*DASK_CHUNK_DIMS*

By default image data are read as `ZYX` chunks, meaning that a single flocal plane is read to memory at a time. This offers maximal potential to scale out to distribute the large datasets but can impact overall performance. For relatively small image files, you will likely get better performance by defining a different strategy to chunk the data. For example, if your dataset has a single focal plane (Z=1), the timeseries analysis will likely perform faster using `TYX` chunking.

You can set the chunking strategy via environment variable like so:
```
export DASK_CHUNK_DIMS=TYX
```

*DASK_PERSIST_INPUT*

When `DASK_PERSIST_INPUT` is set to `1` (or `"True"`), image dask arrays are persisted in memory. This can provide a significant performance boost for small image arrays. However, large image arrays may produce an out-of-memory error. 

For example:
```
DASK_PERSIST_INPUT=1
```

*DASK_PERFORMANCE_REPORT*

When `DASK_PERFORMANCE_REPORT` is specified, the dask scheduler will write a performance report in html format to this location. The file location must be writeable by the dask scheduler.

For example:
```
DASK_PERFORMANCE_REPORT="/MY/PATH/dask-report.html"
```


## Contributing

Contributions are very welcome.

## License

Distributed under the terms of the [BSD-3] license, "atracks" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[BSD-3]: http://opensource.org/licenses/BSD-3-Clause

[file an issue]: https://github.com/uvarc/atracks/issues

[pip]: https://pypi.org/project/pip/

[PyPI]: https://pypi.org/


