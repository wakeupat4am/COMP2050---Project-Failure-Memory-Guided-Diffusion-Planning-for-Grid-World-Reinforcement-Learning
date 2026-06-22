# Cross-Map Deployment Experiment

This folder adds a deployment-style experiment without changing the existing project or `report_package`.

## Goal

Train one **map-conditioned** diffusion model across many generated GridWorld maps, then deploy the resulting model on **one held-out unseen map**. The diffusion-family planners reuse the existing planning logic, but the diffusion condition is extended from:

- current state
- goal state

to:

- flattened obstacle map
- current state
- goal state

This makes cross-map transfer a real test instead of forcing the model to infer unseen obstacle layouts only through simulator collisions.

## What Is Reused

This experiment reuses the existing project where it is already clean:

- `GridWorldEnv`
- BFS helpers
- evaluation helpers
- seed/statistics helpers
- Value Iteration and MCTS
- the existing Failure-Memory and Improved Failure-Memory planner logic

The new code inside this folder only adds:

- random multi-map deployment data generation
- map-conditioned diffusion training data construction
- map-conditioned planner subclasses
- held-out deployment evaluation
- local result figures and tables

## Protocol

1. Generate many training maps of one fixed grid size.
2. Sample several reachable goals per training map.
3. Build BFS training trajectories from every reachable free start state to each sampled goal.
4. Train one map-conditioned diffusion model.
5. Generate one unseen hold-out map.
6. Sample multiple start-goal deployment tasks on that hold-out map.
7. Evaluate:
   - `Random Policy`
   - `Value Iteration`
   - `Policy Iteration`
   - `Q-learning`
   - `MCTS`
   - `Map-Conditioned Standard Diffusion`
   - `Map-Conditioned Failure-Memory Diffusion`
   - `Map-Conditioned Improved Failure-Memory Diffusion`

## Run

```bash
python3 application_deployed/run_cross_map_deployment.py
```

Optional example with a lighter configuration:

```bash
python3 application_deployed/run_cross_map_deployment.py --num-training-maps 12 --num-holdout-tasks 8 --seeds 0 1
```

## Outputs

Generated outputs are written under:

- `application_deployed/results/tables/`
- `application_deployed/results/figures/`
- `application_deployed/results/RESULTS.md`

Key files:

- `deployment_summary.csv`: final held-out efficiency summary across seeds
- `deployment_per_seed.csv`: per-seed held-out results
- `training_loss.csv`: diffusion training loss by epoch and seed
- `heldout_map_overview.png`: unseen deployment map with task markers
- `deployment_efficiency_dashboard.png`: success, collision, return, and inference-time comparison
- `training_loss_curve.png`: training loss across epochs

## Notes

- The folder is intentionally self-contained and does not modify the existing benchmark pipeline.
- The diffusion model is still trained from BFS trajectories, but BFS is only used to create offline supervision.
- During deployment, the diffusion-family planners still score sampled candidates with the environment simulator, consistent with the rest of the project.
