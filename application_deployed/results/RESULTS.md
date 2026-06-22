# Deployment Experiment Summary

- Training maps: `18`
- Training samples: `1362`
- Model seeds evaluated: `3`
- Held-out map free cells: `26`
- Held-out tasks: `10`

## Held-Out Efficiency

| Algorithm | Success Rate | Collision Rate | Average Return | Inference Time |
| --- | --- | --- | --- | --- |
| MCTS | 0.913 +- 0.012 | 0.000 +- 0.000 | 0.732 +- 0.023 | 1.436 ms +- 0.040 ms |
| Map-Conditioned Failure-Memory Diffusion | 0.280 +- 0.020 | 0.720 +- 0.020 | -0.484 +- 0.037 | 1.417 ms +- 0.008 ms |
| Map-Conditioned Improved Failure-Memory Diffusion | 1.000 +- 0.000 | 0.000 +- 0.000 | 0.899 +- 0.008 | 1.995 ms +- 0.053 ms |
| Map-Conditioned Standard Diffusion | 0.133 +- 0.031 | 0.867 +- 0.031 | -0.763 +- 0.059 | 1.358 ms +- 0.004 ms |
| Policy Iteration | 1.000 +- 0.000 | 0.000 +- 0.000 | 0.940 +- 0.000 | 0.000 ms +- 0.000 ms |
| Q-learning | 1.000 +- 0.000 | 0.000 +- 0.000 | 0.940 +- 0.000 | 0.001 ms +- 0.000 ms |
| Random Policy | 0.000 +- 0.000 | 1.000 +- 0.000 | -1.015 +- 0.003 | 0.001 ms +- 0.000 ms |
| Value Iteration | 1.000 +- 0.000 | 0.000 +- 0.000 | 0.940 +- 0.000 | 0.000 ms +- 0.000 ms |