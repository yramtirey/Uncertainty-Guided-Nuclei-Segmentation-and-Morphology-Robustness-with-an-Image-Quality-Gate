# Results summary

The saved output files in [outputs/histories](../outputs/histories) are the canonical project result tables.

## Final comparison from saved outputs

| Dataset | Passive Dice | Active Dice | Passive IoU | Active IoU |
| --- | ---: | ---: | ---: | ---: |
| BBBC006 | 0.947 | 0.948 | 0.907 | 0.909 |
| BBBC039 | 0.956 | 0.957 | 0.916 | 0.918 |
| combined | 0.573 | 0.623 | 0.502 | 0.547 |

## Interpretation

These values are the recorded metrics from the downstream summary in the project outputs. The combined dataset shows the largest active-learning gain, while BBBC006 and BBBC039 show small gains at the same fixed labeling budget.

## Additional artifacts

- Model checkpoints are saved in [outputs/checkpoints](../outputs/checkpoints)
- Training histories are saved in [outputs/histories](../outputs/histories)
- Selected example feature summaries are stored in [outputs/histories](../outputs/histories)
- Visualization figures are stored in [outputs/figures](../outputs/figures)
