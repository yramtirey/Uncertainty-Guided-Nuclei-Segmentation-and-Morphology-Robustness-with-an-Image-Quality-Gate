# Limitations and scope

This project should be interpreted as a final project snapshot for a segmentation experiment, not as a general-purpose benchmark for all microscopy segmentation workflows.

## Scope of the evidence

The saved outputs support the specific comparison setup represented in the notebooks and CSV summaries:

- BBBC006, BBBC039, and combined-dataset active/passive comparison
- a matched labeling budget of 120 labeled samples
- a warm-up + MC dropout uncertainty selection strategy
- repeated training under the recorded setup used in the project

These results do not automatically generalize to other imaging domains, larger budgets, different segmentation tasks, or alternative architectures.

## Reproducibility caveat

The original notebooks were created in a Google Drive/Colab environment and include absolute path assumptions. This repository preserves the final science while moving toward a cleaner local workflow, but some notebook cells may still need path adjustment if they are run in a different environment.

## Statistical caution

The project reports the numbers found in the saved outputs, and those values are treated as the project’s recorded result set. A small number of experiments are not enough to claim broad superiority across all seeds, datasets, or labeling regimes.

## Practical interpretation

The most defensible conclusion from the repository is that, under the recorded setup, the MC dropout active-learning strategy produced a small gain on the BBBC006 and BBBC039 comparisons and a larger gain on the combined dataset comparison within that fixed experimental design.
