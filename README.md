# Failure-Memory Guided Diffusion Planning for Grid-World Reinforcement Learning

This project implements a lightweight GridWorld coursework pipeline for comparing planning and reinforcement learning methods, with special focus on standard diffusion planning versus failure-memory guided diffusion planning.

## Folder Structure

```text
failure_memory_diffusion_project/
    README.md
    requirements.txt
    main.py

    envs/
        __init__.py
        gridworld.py

    planners/
        __init__.py
        random_policy.py
        value_iteration.py
        policy_iteration.py
        q_learning.py
        mcts.py
        diffusion_model.py
        diffusion_planner.py
        failure_memory_planner.py

    utils/
        __init__.py
        bfs.py
        metrics.py
        map_generator.py
        plotting.py
        seed.py

    experiments/
        __init__.py
        run_comparison.py

    results/
        tables/
        figures/
```

## Installation

```bash
pip install -r requirements.txt
```

## How To Run

From inside `failure_memory_diffusion_project/`:

```bash
python main.py
```

## Implemented Algorithms

- Random Policy: chooses a random action at each step.
- Value Iteration: model-based oracle planner using Bellman optimality updates.
- Policy Iteration: model-based oracle planner using policy evaluation and improvement.
- Q-learning: tabular model-free reinforcement learning baseline.
- MCTS: online tree-search planner with UCB selection and heuristic rollouts.
- Standard Diffusion Planner: generates candidate action sequences with a conditional diffusion model and scores them by simulated return and distance-to-goal.
- Failure-Memory Guided Diffusion Planner: extends standard diffusion planning with a failure-memory penalty over risky states seen in failed episodes.

## Output

Running the project produces:

- `results/tables/efficiency_comparison.csv`
- a printed pandas DataFrame in the terminal
- optional figures in `results/figures/`

The final table reports success, return, path length, collision rate, optimality gap, repeated failure rate, and inference time for every algorithm-map combination.
