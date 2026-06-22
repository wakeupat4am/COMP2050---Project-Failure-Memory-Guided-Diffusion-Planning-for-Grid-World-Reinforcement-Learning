from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "report_package"
FIGURES_DIR = OUTPUT_DIR / "figures"


def fmt(value: float, digits: int = 3) -> str:
    if pd.isna(value):
        return "NA"
    return f"{value:.{digits}f}"


def mean_std(row: pd.Series, metric: str, digits: int = 3) -> str:
    return f"{fmt(row[f'{metric} Mean'], digits)} +- {fmt(row[f'{metric} Std'], digits)}"


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    header_line = "| " + " | ".join(headers) + " |"
    divider = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_line, divider, *body])


def copy_figures() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    figure_map = {
        BASE_DIR / "final_image" / "map_scenes.png": "01_map_scenes.png",
        BASE_DIR / "final_image" / "planner_comparison.png": "02_planner_comparison.png",
        BASE_DIR / "Exploration" / "benchmark_results" / "figures" / "success_rate_comparison.png": "03_success_rate_comparison.png",
        BASE_DIR / "Exploration" / "benchmark_results" / "figures" / "average_return_comparison.png": "04_average_return_comparison.png",
        BASE_DIR / "Exploration" / "benchmark_results" / "figures" / "collision_rate_comparison.png": "05_collision_rate_comparison.png",
        BASE_DIR / "Exploration" / "benchmark_results" / "figures" / "inference_time_comparison.png": "06_inference_time_comparison.png",
        BASE_DIR / "Exploration" / "benchmark_results" / "figures" / "diffusion_focus_comparison.png": "07_diffusion_focus_comparison.png",
        BASE_DIR / "Exploration" / "results" / "figures" / "exploration_success_rate.png": "08_exploration_success_rate.png",
        BASE_DIR / "Exploration" / "component_ablation_results" / "figures" / "component_ablation_success_rate.png": "09_component_ablation_success_rate.png",
        BASE_DIR / "results" / "figures" / "lambda_failure_ablation.png": "10_lambda_failure_ablation.png",
        BASE_DIR / "results" / "figures" / "k_ablation_failure_memory.png": "11_k_ablation_failure_memory.png",
        BASE_DIR.parent / "application_deployed" / "results" / "figures" / "deployment_efficiency_dashboard.png": "12_deployment_efficiency_dashboard.png",
        BASE_DIR.parent / "application_deployed" / "results" / "figures" / "heldout_map_overview.png": "13_heldout_map_overview.png",
    }
    for source, target_name in figure_map.items():
        shutil.copy2(source, FIGURES_DIR / target_name)


def build_report() -> str:
    benchmark = pd.read_csv(BASE_DIR / "Exploration" / "benchmark_results" / "tables" / "improved_benchmark_comparison.csv")
    benchmark_seed = pd.read_csv(BASE_DIR / "Exploration" / "benchmark_results" / "tables" / "improved_benchmark_comparison_per_seed.csv")
    exploration = pd.read_csv(BASE_DIR / "Exploration" / "results" / "tables" / "exploration_comparison.csv")
    component = pd.read_csv(BASE_DIR / "Exploration" / "component_ablation_results" / "tables" / "component_ablation.csv")
    lambda_f = pd.read_csv(BASE_DIR / "results" / "tables" / "failure_memory_lambda_ablation.csv")
    k_f = pd.read_csv(BASE_DIR / "results" / "tables" / "failure_memory_k_ablation.csv")
    deployment_dir = BASE_DIR.parent / "application_deployed" / "results" / "tables"
    deployment_summary = pd.read_csv(deployment_dir / "deployment_summary.csv")
    holdout_tasks = pd.read_csv(deployment_dir / "holdout_tasks.csv")
    training_samples = pd.read_csv(deployment_dir / "training_samples.csv")
    with (deployment_dir / "deployment_config.json").open("r", encoding="utf-8") as handle:
        deployment_config = json.load(handle)
    seed_count = int(benchmark["Seed Count"].max())

    difficult_maps = benchmark[
        benchmark["Algorithm"].isin(
            [
                "Standard Diffusion",
                "Failure-Memory Diffusion",
                "Improved Failure-Memory Diffusion",
            ]
        )
        & benchmark["Map"].isin(["obstacle", "deceptive"])
    ].copy()
    diffusion_rows = []
    for _, row in difficult_maps.sort_values(["Map", "Algorithm"]).iterrows():
        diffusion_rows.append(
            [
                row["Algorithm"],
                row["Map"],
                mean_std(row, "Success Rate"),
                mean_std(row, "Collision Rate"),
                mean_std(row, "Average Return"),
            ]
        )
    diffusion_table = markdown_table(
        ["Planner", "Map", "Success Rate", "Collision Rate", "Average Return"],
        diffusion_rows,
    )

    deceptive_seed = benchmark_seed[
        benchmark_seed["Algorithm"].isin(["Failure-Memory Diffusion", "Improved Failure-Memory Diffusion"])
        & (benchmark_seed["Map"] == "deceptive")
    ]
    deceptive_seed_ids = sorted(int(seed) for seed in deceptive_seed["Seed"].unique())
    seed_rows = []
    for algorithm in ["Failure-Memory Diffusion", "Improved Failure-Memory Diffusion"]:
        subset = deceptive_seed[deceptive_seed["Algorithm"] == algorithm].sort_values("Seed")
        success_values = subset["Success Rate"].tolist()
        collision_values = subset["Collision Rate"].tolist()
        repeat_values = subset["Repeated Failure Rate"].tolist()
        seed_rows.append(
            [
                algorithm,
                ", ".join(
                    f"s{seed_id}={fmt(success_value)}"
                    for seed_id, success_value in zip(deceptive_seed_ids, success_values)
                ),
                f"{fmt(sum(success_values) / len(success_values))} +- {fmt(pd.Series(success_values).std(ddof=1))}",
                f"{fmt(sum(collision_values) / len(collision_values))} +- {fmt(pd.Series(collision_values).std(ddof=1))}",
                f"{fmt(sum(repeat_values) / len(repeat_values))} +- {fmt(pd.Series(repeat_values).std(ddof=1))}",
            ]
        )
    seed_table = markdown_table(
        [
            "Planner",
            "Per-seed success values",
            "Success Mean +- SD",
            "Collision Mean +- SD",
            "Repeat Mean +- SD",
        ],
        seed_rows,
    )

    exploration_rows = []
    deceptive_explore = exploration[exploration["Map"] == "deceptive"].copy().sort_values("Algorithm")
    for _, row in deceptive_explore.iterrows():
        exploration_rows.append(
            [
                row["Algorithm"],
                mean_std(row, "Success Rate"),
                mean_std(row, "Collision Rate"),
                mean_std(row, "Average Return"),
                mean_std(row, "Repeated Failure Rate"),
            ]
        )
    exploration_table = markdown_table(
        ["Variant", "Success Rate", "Collision Rate", "Average Return", "Repeated Failure Rate"],
        exploration_rows,
    )

    component_rows = []
    deceptive_component = component[component["Map"] == "deceptive"].copy().sort_values("Algorithm")
    for _, row in deceptive_component.iterrows():
        component_rows.append(
            [
                row["Algorithm"],
                mean_std(row, "Success Rate"),
                mean_std(row, "Collision Rate"),
                mean_std(row, "Average Return"),
                mean_std(row, "Repeated Failure Rate"),
            ]
        )
    component_table = markdown_table(
        ["Variant", "Success Rate", "Collision Rate", "Average Return", "Repeated Failure Rate"],
        component_rows,
    )

    lambda_rows = []
    for _, row in lambda_f.iterrows():
        lambda_rows.append(
            [
                fmt(row["lambda_F"], 1),
                mean_std(row, "Success Rate"),
                mean_std(row, "Collision Rate"),
                mean_std(row, "Average Return"),
            ]
        )
    lambda_table = markdown_table(
        ["lambda_F", "Success Rate", "Collision Rate", "Average Return"],
        lambda_rows,
    )

    k_rows = []
    for _, row in k_f.iterrows():
        k_rows.append(
            [
                str(int(row["K"])),
                mean_std(row, "Success Rate"),
                mean_std(row, "Collision Rate"),
                mean_std(row, "Repeated Failure Rate"),
                f"{fmt(row['Inference Time Mean'] * 1000, 3)} ms +- {fmt(row['Inference Time Std'] * 1000, 3)} ms",
            ]
        )
    k_table = markdown_table(
        ["K", "Success Rate", "Collision Rate", "Repeated Failure Rate", "Inference Time"],
        k_rows,
    )

    deployment_algorithms = [
        "Random Policy",
        "Value Iteration",
        "Policy Iteration",
        "Q-learning",
        "MCTS",
        "Map-Conditioned Improved Failure-Memory Diffusion",
    ]
    deployment_focus = deployment_summary[
        deployment_summary["Algorithm"].isin(deployment_algorithms)
    ].copy()
    deployment_rows = []
    for algorithm in deployment_algorithms:
        subset = deployment_focus[deployment_focus["Algorithm"] == algorithm]
        if subset.empty:
            continue
        row = subset.iloc[0]
        deployment_rows.append(
            [
                row["Algorithm"],
                mean_std(row, "Success Rate"),
                mean_std(row, "Collision Rate"),
                mean_std(row, "Average Return"),
                f"{fmt(row['Inference Time Mean'] * 1000, 3)} ms +- {fmt(row['Inference Time Std'] * 1000, 3)} ms",
            ]
        )
    deployment_table = markdown_table(
        ["Algorithm", "Success Rate", "Collision Rate", "Average Return", "Inference Time"],
        deployment_rows,
    )

    implementation_table = markdown_table(
        ["Component", "Setting"],
        [
            ["Environment", "Deterministic GridWorld with 4 actions, max_steps = 50"],
            ["Rewards", "goal = +1.0, collision = -1.0, move = -0.01, timeout = -0.5"],
            ["Diffusion input", "12-step horizon, 4-way one-hot action encoding, 48-dim flattened trajectory"],
            ["Benchmark condition", "[row, col, goal_row, goal_col] normalized by grid size"],
            ["Network", "Conditional MLP: 48+4+64 -> 128 -> 128 -> 48"],
            ["Activations", "ReLU"],
            ["Time embedding", "64-dim sinusoidal embedding"],
            ["Optimizer", "Adam, learning rate = 1e-3"],
            ["Batch size", "32 in all rerun experiments"],
            ["Noise schedule", "Linear beta schedule from 1e-4 to 0.02 over 25 diffusion steps"],
            ["Benchmark training data", "BFS shortest paths from every reachable traversable start state on each map"],
            ["Padding rule", "Sequences shorter than the horizon repeat the final action"],
            ["Benchmark retraining policy", "One diffusion model retrained for each seed and each map"],
            ["Deployment condition", "Flattened occupancy grid plus normalized [row, col, goal_row, goal_col]"],
            ["Deployment training protocol", "One map-conditioned diffusion model trained across many generated maps and evaluated on one unseen hold-out map"],
            ["Standard diffusion", "lambda_D = 0.1, K selected from {5, 10, 20, 30, 40}"],
            ["Failure-memory diffusion", "lambda_F sweep over {0.0, 0.1, 0.5, 1.0, 2.0, 5.0}; K sweep over {5, 10, 20, 30, 40}"],
            ["Improved planner", "tail_k = 5, raw_multiplier = 3, lambda_loop = 0.2"],
            ["Adaptive weights", "lambda_D(t) = lambda_D,0 / (1 + alpha * F_recent), lambda_F(t) = lambda_F,0 * (1 + beta * F_recent)"],
            ["Adaptive parameters", "lambda_D,0 = 0.1, lambda_F,0 = 0.5, alpha = 0.5, beta = 0.5, window W = 10"],
            ["Q-learning", "alpha = 0.1, gamma = 0.99, epsilon = 1.0 -> 0.05 with decay 0.995, 400 training episodes"],
            ["MCTS", "25 simulations/action, UCB exploration constant 1.41, heuristic rollout enabled, rollout depth 30"],
            ["Value/Policy Iteration", "gamma = 0.99, convergence threshold theta = 1e-6"],
            ["Timing hardware", "Apple M4 MacBook Air, 10-core CPU, 16 GB RAM, torch 2.9.0 CPU-only"],
        ],
    )

    improved_deceptive = benchmark[
        (benchmark["Algorithm"] == "Improved Failure-Memory Diffusion") & (benchmark["Map"] == "deceptive")
    ].iloc[0]
    failure_deceptive = benchmark[
        (benchmark["Algorithm"] == "Failure-Memory Diffusion") & (benchmark["Map"] == "deceptive")
    ].iloc[0]
    improved_obstacle = benchmark[
        (benchmark["Algorithm"] == "Improved Failure-Memory Diffusion") & (benchmark["Map"] == "obstacle")
    ].iloc[0]
    without_adaptive = component[
        (component["Algorithm"] == "Without Adaptive Weights") & (component["Map"] == "deceptive")
    ].iloc[0]
    without_tail = component[
        (component["Algorithm"] == "Without Tail Memory") & (component["Map"] == "deceptive")
    ].iloc[0]
    obstacle_improved_seed = benchmark_seed[
        (benchmark_seed["Algorithm"] == "Improved Failure-Memory Diffusion") & (benchmark_seed["Map"] == "obstacle")
    ].sort_values("Seed")
    improved_deceptive_seed = deceptive_seed[deceptive_seed["Algorithm"] == "Improved Failure-Memory Diffusion"].sort_values("Seed")
    failure_deceptive_seed = deceptive_seed[deceptive_seed["Algorithm"] == "Failure-Memory Diffusion"].sort_values("Seed")
    per_seed_success_improvement = (
        improved_deceptive_seed["Success Rate"].to_numpy() - failure_deceptive_seed["Success Rate"].to_numpy()
    )
    improvement_seed_wins = int((per_seed_success_improvement > 0).sum())
    improvement_seed_non_losses = int((per_seed_success_improvement >= 0).sum())
    success_consistency_phrase = (
        f"was consistent across all {seed_count} seeds"
        if improvement_seed_wins == seed_count
        else f"appeared in {improvement_seed_wins} of {seed_count} seeds"
    )
    per_seed_comparison_phrase = (
        f"was strictly better in all {seed_count} seeds"
        if improvement_seed_wins == seed_count
        else f"matched or exceeded the original failure-memory planner in all {improvement_seed_non_losses} seeds, with strict improvement in {improvement_seed_wins} of {seed_count}"
    )
    zero_collision_all_difficult = bool(
        (improved_deceptive_seed["Collision Rate"] == 0).all() and (obstacle_improved_seed["Collision Rate"] == 0).all()
    )
    zero_collision_phrase = (
        f"collision dropped to zero across all {seed_count} seeds"
        if zero_collision_all_difficult
        else "collision was reduced strongly, though not to zero in every seed"
    )
    k5_row = k_f.loc[k_f["K"] == 5].iloc[0]
    k40_row = k_f.loc[k_f["K"] == 40].iloc[0]
    deployment_improved = deployment_summary[
        deployment_summary["Algorithm"] == "Map-Conditioned Improved Failure-Memory Diffusion"
    ].iloc[0]
    deployment_mcts = deployment_summary[
        deployment_summary["Algorithm"] == "MCTS"
    ].iloc[0]
    deployment_qlearning = deployment_summary[
        deployment_summary["Algorithm"] == "Q-learning"
    ].iloc[0]
    deployment_value_iteration = deployment_summary[
        deployment_summary["Algorithm"] == "Value Iteration"
    ].iloc[0]
    deployment_training_map_count = int(deployment_config["num_training_maps"])
    deployment_model_seed_count = int(len(deployment_config["model_seeds"]))
    deployment_holdout_task_count = int(len(holdout_tasks))
    deployment_training_sample_count = int(len(training_samples))

    report = f"""# Improving Diffusion-Based GridWorld Planning with Online Failure Memory

## Package Layout
- `REPORT.md`: revised report with updated statistics, clearer claims, and figure notes.
- `figures/`: curated figure set copied from the refreshed experiment outputs.

## Abstract
This revised report studies diffusion-based planning in a deterministic GridWorld and focuses on a more defensible evaluation protocol than the earlier draft. The diffusion model is trained on BFS shortest-path demonstrations and then used online to propose candidate trajectories. The main research question is whether online failure memory can improve candidate ranking without retraining the generator.

The key methodological upgrades in this version are: seed-level reporting, 95% confidence intervals over seed means in the figures, numeric parameter sweeps for `lambda_F` and `K`, a new remove-one-component ablation for the final planner, and a deployment-style cross-map study that trains one map-conditioned improved planner across many maps before evaluating it on one unseen hold-out map. Under the refreshed benchmark, Improved Failure-Memory Diffusion reached `success = {fmt(improved_obstacle['Success Rate Mean'])} +- {fmt(improved_obstacle['Success Rate Std'])}` with zero collision on the obstacle map and `success = {fmt(improved_deceptive['Success Rate Mean'])} +- {fmt(improved_deceptive['Success Rate Std'])}` with zero collision on the deceptive map. The deceptive-map gain over the original failure-memory planner {success_consistency_phrase}. In the deployment study, the map-conditioned improved planner achieved `success = {fmt(deployment_improved['Success Rate Mean'])} +- {fmt(deployment_improved['Success Rate Std'])}` on a held-out unseen map, though with higher inference time than the classical baselines.

## What Changed Relative to the Earlier Draft
1. All main reported metrics are now aggregated per seed first and summarized as `mean +- SD` across the {seed_count} seed means.
2. Main comparison and ablation figures now show 95% confidence intervals over seed means.
3. The qualitative candidate-budget summary was replaced with a numeric `K` sweep.
4. The `lambda_F` tuning discussion is now tied to an explicit sweep table and figure.
5. A new remove-one-component ablation was added for the final planner.
6. Reproducibility details are collected in one implementation table.
7. The report framing now separates oracle/reference baselines from approximate and learned planners.
8. A deployment-style cross-map generalization study was added to test whether the improved planner can be carried toward more realistic use.

## Problem Setup and Fairness Framing
The task is a deterministic, fully observable GridWorld with four actions, fixed start and goal states, collision termination, and a 50-step limit. The comparison mixes methods with different assumptions, so the baselines should be interpreted in two tiers:

- `Value Iteration` and `Policy Iteration` are reference baselines with full transition knowledge.
- `Q-learning`, `MCTS`, and the diffusion-family planners are approximate methods with different training and inference budgets.

This report does not claim that the diffusion planner is more general or cheaper than all baselines. The narrower goal is to show whether online failure-aware ranking improves the diffusion-family planner under the same learned generator.

## Related Work and Research Gap
Diffusion planners such as Diffuser, Decision Diffuser, and later action-diffusion policies show that generative trajectory models can represent multimodal action sequences. However, these methods typically guide trajectories through rewards, task constraints, or learned value estimates. They do not usually maintain an explicit online spatial memory of repeated evaluation failures during execution.

The research gap for this project is therefore narrower than general reinforcement learning: can online failure memory change candidate ranking enough to improve a fixed diffusion generator without retraining it after each failed episode?

## Research Questions
This report is framed as a controlled study of diffusion-planner behavior in deterministic GridWorld navigation rather than only as a leaderboard comparison. The evaluation is organized around five research questions:

1. How does a diffusion-based planner behave in small deterministic GridWorlds when it is trained on BFS shortest-path demonstrations?
2. Can online failure memory improve the robustness of a fixed diffusion planner on obstacle-heavy and deceptive maps without retraining the generator?
3. Which components of the final planner, tail-focused memory, adaptive weighting, candidate diversity, and loop penalty, contribute most to the observed improvement?
4. How do the candidate budget `K` and the failure-memory coefficient `lambda_F` affect the trade-off between success, safety, and inference cost?
5. Can the improved planner be adapted into a more deployment-realistic cross-map setting, and what advantages and limitations appear when it is compared against classical planners on an unseen map?

The benchmark section answers Research Questions 1 and 2, the exploration and remove-one ablations answer Research Question 3, the parameter sweeps answer Research Question 4, and the deployment study answers Research Question 5.

## Method Summary
The final planner combines four mechanisms:
1. Tail-only failure-memory updates.
2. Adaptive distance and failure penalties based on recent failures.
3. First-action diversity through oversampling and regrouping.
4. A repeated-state loop penalty.

The earlier large box-diagram figures were not reused in this package because they did not scale reliably for double-column layout. Instead, the package keeps a smaller conceptual comparison figure and adds clearer notes next to the empirical plots.

![Evaluation maps](figures/01_map_scenes.png)

*Figure note.* The benchmark uses four maps: an easy open map, an obstacle map, a deceptive corridor map, and one random small map. The deceptive map is the main stress test because the shortest successful route initially moves away from the goal.

![Planner comparison](figures/02_planner_comparison.png)

*Figure note.* The three diffusion-family planners share the same generator but differ in candidate scoring and memory handling. This makes the family comparison more meaningful than comparing raw success rates against exact planners alone.

## Reproducibility and Implementation Details
{implementation_table}

## Evaluation Protocol and Statistical Reporting
- Seeds: `{", ".join(str(seed) for seed in sorted(benchmark_seed["Seed"].unique()))}`
- Evaluation episodes per seed: `10`
- Main benchmark total per algorithm-map combination: `{seed_count * 10}` episodes
- Reported tables: seed means summarized as `mean +- SD`
- Figures: error bars show 95% confidence intervals over the {seed_count} seed means
- Even with {seed_count} seeds, the report emphasizes uncertainty and effect direction rather than overstating significance

Raw, per-seed, and summary CSV files were regenerated for the benchmark, exploration study, `lambda_F` sweep, `K` sweep, and component ablation. These refreshed outputs now live in:
- `Exploration/benchmark_results/tables/`
- `Exploration/results/tables/`
- `Exploration/component_ablation_results/tables/`
- `results/tables/`

## Main Benchmark
The strongest evidence for the improved planner comes from the two difficult maps:

{diffusion_table}

The deceptive-map improvement is not a single pooled proportion hiding seed instability. The per-seed summary below shows that the improved planner {per_seed_comparison_phrase}:

{seed_table}

![Success-rate comparison](figures/03_success_rate_comparison.png)

*Figure note.* Error bars show 95% confidence intervals over seed means. `Value Iteration` and `Policy Iteration` should be read as reference ceilings rather than budget-matched competitors.

![Average-return comparison](figures/04_average_return_comparison.png)

*Figure note.* The deceptive map remains difficult even when success improves, which is why average return remains below zero for the diffusion-family planners on that map.

![Collision-rate comparison](figures/05_collision_rate_comparison.png)

*Figure note.* The main robustness gain from the improved planner is collision suppression: on both obstacle and deceptive maps, {zero_collision_phrase}.

![Inference-time comparison](figures/06_inference_time_comparison.png)

*Figure note.* Timing was measured as mean action-selection time on a CPU-only Apple M4 laptop. The improved planner is more expensive than the standard and original failure-memory variants because it oversamples and re-ranks more trajectories.

![Diffusion-family comparison](figures/07_diffusion_focus_comparison.png)

*Figure note.* This plot isolates the diffusion-family planners and makes the uncertainty easier to inspect than in the all-algorithm figure.

### Main Benchmark Interpretation
The revised benchmark supports three careful conclusions:

1. The improved planner clearly dominates the original failure-memory planner on the deceptive map. Its success rate increased from `{fmt(failure_deceptive['Success Rate Mean'])} +- {fmt(failure_deceptive['Success Rate Std'])}` to `{fmt(improved_deceptive['Success Rate Mean'])} +- {fmt(improved_deceptive['Success Rate Std'])}`, and collision fell from `{fmt(failure_deceptive['Collision Rate Mean'])} +- {fmt(failure_deceptive['Collision Rate Std'])}` to zero.
2. The obstacle map result is even stronger: the improved planner reached perfect success with zero collision across all seeds.
3. Exact planners still outperform the diffusion-family methods on path efficiency and computational cost, so the contribution is best framed as a within-family improvement rather than a universal planner replacement.

## Exploration Study
The earlier exploration study remains useful because it separates the larger design ideas before the final combined method:

{exploration_table}

![Exploration success-rate comparison](figures/08_exploration_success_rate.png)

*Figure note.* The deceptive map is the main differentiator. The `Combined Exploration` planner achieved the best trade-off among the pre-final variants, while `Adaptive Failure` was the only single enhancement that substantially improved success on its own.

### Exploration Interpretation
The deceptive-map exploration table shows that adaptive weighting was the most important single ingredient before the final combination. Diversity alone removed collisions but did not produce a large success gain, while dead-end memory alone remained close to the original baseline.

## Remove-One Component Ablation
The refreshed report now includes the ablation that was missing in the earlier draft:

{component_table}

![Component ablation success rates](figures/09_component_ablation_success_rate.png)

*Figure note.* This figure compares the full planner against variants where exactly one component is removed. Error bars again show 95% confidence intervals over seed means.

### Component Interpretation
This ablation changes the earlier story in an important way:

1. Removing adaptive weights caused the deceptive-map success rate to collapse from `{fmt(improved_deceptive['Success Rate Mean'])} +- {fmt(improved_deceptive['Success Rate Std'])}` to `{fmt(without_adaptive['Success Rate Mean'])} +- {fmt(without_adaptive['Success Rate Std'])}`. Adaptive weighting is therefore the most critical component in the final planner.
2. Removing diversity or the loop penalty had little effect under the current budget on the deceptive map. These mechanisms may still help stability, but this rerun does not justify strong claims that they are individually decisive.
3. Removing tail-only memory unexpectedly increased deceptive-map success to `{fmt(without_tail['Success Rate Mean'])} +- {fmt(without_tail['Success Rate Std'])}` while also increasing the repeated-failure rate from `{fmt(improved_deceptive['Repeated Failure Rate Mean'])} +- {fmt(improved_deceptive['Repeated Failure Rate Std'])}` to `{fmt(without_tail['Repeated Failure Rate Mean'])} +- {fmt(without_tail['Repeated Failure Rate Std'])}`. The revised report should therefore present tail-only memory as a cleaner credit-assignment choice, not as an unconditional empirical improvement.

## Parameter Sweeps
The earlier qualitative discussion of `lambda_F` and `K` is now replaced by numeric tables.

### Failure-Memory Weight Sweep
{lambda_table}

![Failure-memory strength sweep](figures/10_lambda_failure_ablation.png)

*Figure note.* The report should describe `lambda_F = 0.5` as the best value among the tested sweep values, not as a generally optimal setting. The differences are moderate, and the uncertainty bars overlap substantially.

### Candidate-Budget Sweep
{k_table}

![Candidate-budget sweep](figures/11_k_ablation_failure_memory.png)

*Figure note.* Increasing `K` improves success and lowers collision and repeated-failure rate, but it also increases action-selection time from roughly `{fmt(k5_row['Inference Time Mean'] * 1000, 3)} ms` at `K=5` to `{fmt(k40_row['Inference Time Mean'] * 1000, 3)} ms` at `K=40`.

### Sweep Interpretation
The numeric sweeps support a more careful claim than the earlier draft:
- `lambda_F = 0.5` was the strongest setting in this exploratory sweep, but the report should not describe it as independently validated.
- Larger `K` improves robustness under the current generator, but the improvement from `K=30` to `K=40` is smaller than the jump from `K=10` to `K=20`, while the computational cost keeps rising.

## Choosing the Final Planner
The development-stage experiments justify carrying only the improved planner into the deployment study. Relative to the standard diffusion and original failure-memory variants, the improved method was consistently the safest diffusion-family planner on the difficult maps, and it produced the clearest success gains under deceptive geometry. The trade-off is inference cost: oversampling, diversity filtering, adaptive weighting, and loop-aware rescoring make the improved planner slower than the simpler diffusion baselines.

That trade-off is acceptable for the deployment study because the next question is no longer which diffusion-family variant wins on a fixed map, but whether the strongest diffusion-family variant can be moved toward a more realistic application setting. For that reason, the deployment section drops the two weaker diffusion baselines from the main figure and keeps only the improved planner against classical comparison methods.

## Cross-Map Deployment Study
The deployment study moves beyond single-map training and evaluation. Instead of retraining one diffusion model per map, it trains one map-conditioned diffusion model across `{deployment_training_map_count}` procedurally generated training maps with `{deployment_training_sample_count}` BFS-generated supervision samples, then evaluates the learned policy family on one held-out unseen map with `{deployment_holdout_task_count}` start-goal tasks.

Three design changes make this setting more realistic than the fixed-map benchmark:
1. The diffusion model is conditioned on the obstacle layout as well as the current state and goal, so candidate generation can adapt to different map geometry instead of assuming one fixed wall pattern.
2. Training data comes from many maps and many goals rather than one environment, which turns the learned model into a transferable action prior instead of a single-map imitation model.
3. Deployment evaluation uses one unseen hold-out map and compares the improved planner directly with `Random Policy`, exact planning, `Q-learning`, and `MCTS`, rather than only with weaker diffusion variants.

The hold-out map was still drawn from the same procedural family as the training maps, so this is same-distribution cross-map generalization rather than fully out-of-distribution transfer. Also, BFS is not used to control the improved planner on the hold-out map; it is used only offline to create the training targets and to compute reference shortest-path metrics.

![Held-out deployment map](figures/13_heldout_map_overview.png)

*Figure note.* The deployment study trains on many procedurally generated maps and evaluates on one unseen map with multiple start-goal tasks. This is a stronger test of transfer than the fixed-map benchmark, but it still stays within the same map generator family.

The held-out efficiency comparison is:

{deployment_table}

![Deployment efficiency dashboard](figures/12_deployment_efficiency_dashboard.png)

*Figure note.* The deployment figure keeps only the improved diffusion planner from the diffusion family. The two weaker diffusion variants were useful for algorithm development, but they are omitted here because the deployment question is whether the strongest learned planner can compete with classical baselines under a more realistic transfer setup.

### Deployment Interpretation
The deployment result supports a cautious positive answer to Research Question 5. The map-conditioned improved planner reached `success = {fmt(deployment_improved['Success Rate Mean'])} +- {fmt(deployment_improved['Success Rate Std'])}` with zero collision on the held-out unseen map, which is stronger than `MCTS` on success and much stronger than the weaker diffusion-family variants observed during development. This suggests that conditioning on map geometry and training across many maps can convert the improved planner from a fixed-map method into a transferable same-distribution deployment policy.

At the same time, the comparison remains nuanced. `Value Iteration`, `Policy Iteration`, and `Q-learning` each reached `success = {fmt(deployment_value_iteration['Success Rate Mean'])}` in this specific held-out experiment, while the improved deployment planner required more inference time: about `{fmt(deployment_improved['Inference Time Mean'] * 1000, 3)} ms` per action versus `{fmt(deployment_mcts['Inference Time Mean'] * 1000, 3)} ms` for `MCTS` and `{fmt(deployment_qlearning['Inference Time Mean'] * 1000, 3)} ms` for `Q-learning`. The improved planner therefore offers a meaningful transfer advantage over naive diffusion deployment, but not a universal computational advantage over all baselines.

### Advantages and Limitations Relative to Other Algorithms
Advantages of the deployment-oriented improved planner:
- It does not require BFS trajectories on the target hold-out map at deployment time.
- It can transfer across unseen maps from the same generator family because the learned condition includes obstacle layout, state, and goal.
- Its failure-memory and adaptive rescoring logic can suppress unsafe candidate trajectories that raw diffusion alone would still rank too highly.

Limitations relative to the other baselines:
- It still depends on BFS-generated expert supervision during offline training, which is a stronger source of information than reward-only learning.
- It still uses exact simulator scoring online, so it is not a purely feed-forward policy at test time.
- Its inference cost is materially higher than the exact planners and tabular `Q-learning` in this small deterministic setting.
- The hold-out evaluation is same-distribution, not arbitrary real-world transfer, because train and test maps come from the same procedural family.

## Limitations
1. The environment is small, deterministic, and fully observable.
2. In the core benchmark and ablation pipeline, the diffusion model is retrained separately for each seed and each map.
3. Training demonstrations come from BFS with full map knowledge.
4. Online scoring uses the exact simulator, so the diffusion-family planners are model-based at inference time.
5. Even with {seed_count} seeds, statistical power remains moderate compared with a much larger experimental study.
6. The new ablation indicates that some earlier mechanism-level claims were too strong.
7. The deployment study is more realistic than the fixed-map benchmark, but it still evaluates same-distribution transfer within one procedural map generator family.
8. The deployment comparison is not perfectly matched to `Q-learning`, because the improved planner uses expert BFS supervision and simulator-based candidate scoring while tabular `Q-learning` learns from reward interaction.
9. The deployment study still uses only one held-out map, so stronger evidence would require many hold-out maps and broader statistical summaries.

## Future Work
1. Evaluate the deployment pipeline across many held-out maps, map sizes, and obstacle generators instead of one same-distribution hold-out map.
2. Compare against stronger transfer baselines such as map-conditioned reinforcement learning rather than only tabular target-map `Q-learning`.
3. Reduce inference cost through candidate pruning, distillation, or learned value guidance so that the improved planner is more practical under tight latency budgets.
4. Replace exact simulator scoring with a learned world model or partial-lookahead scoring to move closer to realistic real-world planning settings.
5. Extend the setup to partial observability, stochastic transitions, and dynamic obstacles, which are closer to real deployment domains than deterministic static GridWorlds.

## Conclusion
The revised evidence supports a stronger and more honest report because it now answers the research questions directly.

1. Research Question 1 asked how a diffusion-based planner behaves in deterministic GridWorlds when trained on BFS demonstrations. The results show that standard diffusion planning is workable on simple maps but unreliable on deceptive geometry, where it frequently collapses into collision-heavy behavior driven by local goal-distance bias.
2. Research Question 2 asked whether online failure memory improves robustness without retraining the generator. The answer is yes. Relative to the original failure-memory planner, the improved planner consistently increased success on the deceptive and obstacle maps and {zero_collision_phrase} on those two difficult maps.
3. Research Question 3 asked which components matter most. The new ablation shows that adaptive weighting is the most critical mechanism in the final design. Diversity and the loop penalty were not individually decisive under the current budget, and tail-only memory should be described as a cleaner credit-assignment choice rather than a universally stronger empirical variant.
4. Research Question 4 asked how `K` and `lambda_F` control the trade-off between robustness and cost. The sweeps show that `lambda_F = 0.5` was the strongest value among the tested settings in this study, while larger `K` improved success and reduced collision at the cost of slower action selection. The gain from increasing `K` is real but diminishing.
5. Research Question 5 asked whether the improved planner can be pushed toward a more realistic deployment-style application. The answer is partially yes. Training one map-conditioned improved planner across many maps allowed successful transfer to one unseen same-distribution hold-out map without using BFS to control the planner at deployment time. That is a stronger application story than the fixed-map benchmark, but it still relies on expert offline supervision and simulator-based online scoring.

Taken together, these answers support a narrower but more defensible conclusion than the earlier draft. The project does not show that diffusion planning is the best general planner in small known GridWorlds or that it is already ready for unconstrained real-world deployment. It does show that online failure-aware ranking can materially improve diffusion-based planning in deterministic environments, and that the improved version is the right candidate for further study when the goal shifts from fixed-map analysis toward more realistic cross-map application.
"""
    return report


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    copy_figures()
    report = build_report()
    (OUTPUT_DIR / "REPORT.md").write_text(report, encoding="utf-8")
    print(f"Wrote report package to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
