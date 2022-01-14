# GO-Utils
[![CI](https://github.com/IGES-Geospatial/globe-observer-utils/actions/workflows/CI.yml/badge.svg)](https://github.com/IGES-Geospatial/globe-observer-utils/actions/workflows/CI.yml)

This Package is designed to provide utilities for interfacing with GLOBE Observer Data, particularly the Mosquito Habitat Mapper and Landcover Datasets.

## Installation
Run `pip install go-utils` to install this package.

## Contributing

1. [Fork](https://github.com/IGES-Geospatial/globe-observer-utils/fork) this Repo
2. Clone the Repo onto your computer
3. Create a branch (`git checkout -b new-feature`)
4. Make Changes
5. Run necessary quality assurance tools ([Formatter](#Formatter), [Linter](#Linter) ,[Unit Tests](#Unit-Tests), [Documentation](#Previewing-Documentation)).
6. Add your changes (`git commit -am "Commit Message"` or `git add .` followed by `git commit -m "Commit Message"`)
7. Push your changes to the repo (`git push origin new-feature`)
8. Create a pull request

Do note you can locally build the package with `pip install -e .` and run unit tests with `pytest -s go_utils`.

There are also several tools used by this repository to ensure code quality:

### Formatter
This codebase uses the [black formatter](https://github.com/psf/black) to check code format. 

1. Run `pip install black` to get the package.
2. After making changes, run `black ./`.

This will automatically format your code to Python standards.

### Import Sorting
To make sure imports are sorted, [isort](https://github.com/PyCQA/isort) is used.

1. Run `pip install isort` to get the package.
2. After making changes, run `isort ./ --profile black` (the profile black ensures no conflicts with the black formatter)

### Linter
This codebase uses [flake8](https://github.com/pycqa/flake8) to lint code. 

1. Run `pip install flake8` to get the package.
2. After making changes, run `flake8`.

The linter will notify you of any code that isn't up to Python standards.

### Unit Tests
This codebase uses [pytest](https://github.com/pytest-dev/pytest) to run unit tests for the code. 

1. Run `pip install pytest` to get the package. 
2. After making changes, you can run `pytest` to run all unit tests. See [Advanced Usage](#advanced-usage) for more information.

These tests will make sure the code performs as expected.

#### Advanced Usage
To run tests relevant to a specific function/area, there are several markers that can be used:
- `landcover`: tests for Landcover procedures
- `mosquito`: tests for Mosquito Habitat Mapper procedures
- `util`: tests for utility functions
- `downloadtest`: tests for functions that download GLOBE data over the internet.
- `cleanup`: tests for functions involved in the cleanup procedure

To specifically call a subset of tests, the `-m` flag must be used (e.g. `pytest -m "landcover"`). Using the `or` keyword can be used to include multiple subsets (e.g. `pytest -m "landcover or mosquito"`), but do note that the current markers aren't mutually exclusive (the `and` keyword accomplishes this). Using the `not` keyword can be used to exclude subsets (e.g. `pytest -m "not downloadtest"`), this is particularly useful for excluding the download tests as those tend to take a considerable amount of time.

### Previewing Documentation
The documentation for this package is built by the [pdoc module](https://github.com/mitmproxy/pdoc). 

1. Run `pip install pdoc` to get the package.
2. To preview the documentation, run `pdoc -t doc_template --docformat numpy go_utils`.

This will locally host an updated documentation website which lets you preview any changes you may have made to the documentation.
