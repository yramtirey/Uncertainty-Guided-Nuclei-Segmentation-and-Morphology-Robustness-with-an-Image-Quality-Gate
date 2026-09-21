# Reproducibility notes

## Environment

The project is designed to run in a Python environment that includes:

- PyTorch
- NumPy
- pandas
- scikit-learn
- Pillow
- tifffile
- imagecodecs
- matplotlib
- scipy

The repository includes an environment file and a requirements file at the root.

## Execution guidance

1. Create the environment from the root folder.
2. Keep the data in the data/ directory.
3. Run the notebooks from the project root or from the notebooks/ folder.
4. Use the saved outputs under outputs/ as the authoritative comparison artifacts.

## Important caveat

The notebooks were originally written in a Google Drive and Colab workflow. This repository preserves the scientific artifacts but does not guarantee that every notebook will run without updating any absolute-path assumptions in the original environment. The package extraction and compatibility wrapper make it easier to run shared logic from a fresh clone.
