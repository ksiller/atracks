# Atracks

[![License BSD-3](https://img.shields.io/github/license/ksiller/atracks?label=license&style=flat)](https://github.com/ksiller/atracks/blob/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/atracks.svg?color=green)](https://pypi.org/project/atracks)
[![Python Version](https://img.shields.io/pypi/pyversions/atracks.svg?color=green)](https://python.org)
[![codecov](https://codecov.io/gh/ksiller/atracks/branch/main/graph/badge.svg)](https://codecov.io/gh/ksiller/atracks)

Tool to track atom positions.

## Installation 

You can install `atracks` via [pip]:

    pip install atracks


To install latest development version:

    pip install git+https://github.com/ksiller/atracks.git

## Image file formats

Atracks is reading image files using the [Bioio](https://github.com/bioio-devs/bioio) package. A variety of plugins exist to support common image file formats, including .tiff, .ome-tiff, .zarr, .nd2, .czi, .lif, etc.. By installing these additional bioio plugins you can easily expand Atrack's ability to process a large variety of image formats without the need to touch the source code.  

## Running the Atracks application

In a command line shell, run the following command:
```
atracks -i imagestack.nd2 -o my_outputdir -s 1 -d 2 -r 1
```

**Command line arguments:**

```
  -h, --help            show this help message and exit
  -l LOGLEVEL, --loglevel LOGLEVEL
                        logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
  -i INPUT_PATH, --input INPUT_PATH
                        single image file or directory with image files to be processed
  -o OUTPUT_PATH, --output OUTPUT_PATH
                        output file or directory
  -g, --grayscale       convert loaded images/videos to grayscale before processing
  -f FRAMES, --frames FRAMES
                        frames to process; examples: '10' or '2-40' (inclusive). Default: all frames
```                  

## Contributing

Contributions are very welcome.

## License

Distributed under the terms of the [BSD-3] license, "atracks" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[BSD-3]: http://opensource.org/licenses/BSD-3-Clause

[file an issue]: https://github.com/ksiller/atracks/issues

[pip]: https://pypi.org/project/pip/

[PyPI]: https://pypi.org/


