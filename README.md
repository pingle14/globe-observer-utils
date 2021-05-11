# GO-Utils
[![CI](https://github.com/IGES-Geospatial/globe-observer-utils/actions/workflows/CI.yml/badge.svg)](https://github.com/IGES-Geospatial/globe-observer-utils/actions/workflows/CI.yml)

This Package is designed to provide utilities for interfacing with GLOBE Observer Data, particularly the Mosquito Habitat Mapper and Landcover Datasets.

## Installation
Run `pip install go-utils` to install this package.

## Contributing

### Requirements
This codebase uses the black formatter to check code format. Run `pip install black` to get the package. Then after you make changes, run `black ./` before commiting to have your code automatically formatted.

### Contribution Steps
1. [Fork](https://github.com/IGES-Geospatial/globe-observer-utils/fork) this Repo
2. Clone the Repo onto your computer
3. Create a branch (`git checkout -b new-feature`)
4. Make Changes
5. Run Black Formatter (`pip install black` to download and `black ./` to run).
6. Add your changes (`git commit -am "Commit Message"` or `git add .` followed by `git commit -m "Commit Message"`)
7. Push your changes to the repo (`git push origin new-feature`)
8. Create a pull request

Do note you can locally build the package with `pip install -e .` and run unit tests with `pytest -s go_utils`.
