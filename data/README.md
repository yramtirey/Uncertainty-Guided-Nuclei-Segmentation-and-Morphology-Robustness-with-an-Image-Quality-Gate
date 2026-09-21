# Local dataset directory

This directory is the expected local location for the microscopy datasets used in the project experiments.

## Important scope note

The full BBBC006 and BBBC039 datasets are not committed to this GitHub repository.

The project requires the original image and mask data from the Broad Bioimage Benchmark Collection, and those files must be obtained separately before running the notebooks locally.

## Required datasets

The project expects the following data to be available locally:

- BBBC006
- BBBC039
- a combined dataset assembled from the paired image/mask files used in the final experiments

## Local placement

The expected local structure is:

- data/b006/
- data/b039/
- data/combined/

Each folder contains the image and mask files, along with any dataset index or metadata CSV files used by the project notebooks.

## Historical metadata note

Some historical index files in this project originally contained Google Drive paths from the earlier notebook workflow. Those entries are retained only as provenance from the project’s original execution environment and are not part of the intended portable repository workflow.

## Data exclusion

The repository is configured to exclude dataset contents through the project .gitignore rules so that large microscopy image files are not committed to GitHub.

This repository keeps the analysis pipeline, notebooks, saved outputs, and documentation, but not the raw benchmark image sets themselves.
