"""
Command-Line Interface (CLI) for cloud_optimizer using argparse.

Supports subcommands:
- solve:     Generate a workload, solve the MILP, print allocation summary, and optionally export plots.
- benchmark: Run comparative benchmarks (MILP vs. FFD vs. Random Round-Robin) across scales.
- visualize: Generate bin utilization heatmap and Cost vs. Task Drop Pareto frontier plots.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from cloud_optimizer.benchmark import BenchmarkRunner
from cloud_optimizer.generator import WorkloadGenerator
from cloud_optimizer.solver import CloudTaskScheduler
from cloud_optimizer.visualizer import ScheduleVisualizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cloud-optimizer",
        description=(
            "MILP & Multi-Dimensional Knapsack optimizer for scheduling containerized "
            "tasks onto heterogeneous cloud virtual machines."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 1. `solve` subcommand
    solve_parser = subparsers.add_parser(
        "solve", help="Solve a synthetic cloud task scheduling instance via MILP."
    )
    solve_parser.add_argument(
        "-n", "--num-tasks", type=int, default=40, help="Number of tasks N (default: 40)"
    )
    solve_parser.add_argument(
        "-m", "--num-vms", type=int, default=None, help="Number of VMs M (default: auto-sized)"
    )
    solve_parser.add_argument(
        "--budget-tightness",
        type=float,
        default=0.75,
        help="Fraction of total fleet cost allowed as B_max (default: 0.75)",
    )
    solve_parser.add_argument(
        "--lambda-penalty",
        type=float,
        default=5.0,
        help="Penalty weight lambda for unassigned tasks (default: 5.0)",
    )
    solve_parser.add_argument(
        "--time-limit",
        type=float,
        default=20.0,
        help="CBC solver time limit in seconds (default: 20.0)",
    )
    solve_parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    solve_parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to export JSON allocation summary",
    )

    # 2. `benchmark` subcommand
    bench_parser = subparsers.add_parser(
        "benchmark",
        help="Compare MILP against First-Fit Decreasing and Random Round-Robin across scales.",
    )
    bench_parser.add_argument(
        "--scales",
        type=int,
        nargs="+",
        default=[20, 50, 200, 1000],
        help="Task scales N to evaluate (default: 20 50 200 1000)",
    )
    bench_parser.add_argument(
        "--time-limit",
        type=float,
        default=15.0,
        help="Per-instance CBC solver time limit in seconds (default: 15.0)",
    )
    bench_parser.add_argument(
        "--lambda-penalty",
        type=float,
        default=5.0,
        help="Penalty factor lambda (default: 5.0)",
    )
    bench_parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )

    # 3. `visualize` subcommand
    viz_parser = subparsers.add_parser(
        "visualize",
        help="Generate VM bin utilization heatmap and Cost vs. Task Drop Pareto frontier plots.",
    )
    viz_parser.add_argument(
        "-n", "--num-tasks", type=int, default=40, help="Number of tasks N (default: 40)"
    )
    viz_parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts",
        help="Directory to save generated PNG plots (default: artifacts)",
    )
    viz_parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "solve":
        gen = WorkloadGenerator(seed=args.seed)
        config = gen.generate_cluster_config(
            num_tasks=args.num_tasks,
            num_vms=args.num_vms,
            budget_tightness=args.budget_tightness,
            lambda_penalty=args.lambda_penalty,
            solver_time_limit=args.time_limit,
        )
        scheduler = CloudTaskScheduler(config)
        scheduler.build_model()
        scheduler.solve(msg=False)
        summary = scheduler.get_allocation_summary()

        print(json.dumps(summary, indent=2))
        if args.output_json:
            out_path = Path(args.output_json).resolve()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(f"\nSaved allocation summary to: {out_path}")
        return 0

    if args.command == "benchmark":
        runner = BenchmarkRunner(
            scales=args.scales,
            seed=args.seed,
            solver_time_limit=args.time_limit,
            lambda_penalty=args.lambda_penalty,
        )
        rows = runner.run_all()
        print("\n### Comparative Benchmark Results (MILP vs. Heuristics)\n")
        print(runner.format_markdown_table(rows))
        return 0

    if args.command == "visualize":
        out_dir = Path(args.output_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        gen = WorkloadGenerator(seed=args.seed)
        config = gen.generate_cluster_config(
            num_tasks=args.num_tasks,
            budget_tightness=0.75,
            lambda_penalty=5.0,
            solver_time_limit=15.0,
        )
        scheduler = CloudTaskScheduler(config)
        result = scheduler.solve(msg=False)

        heatmap_path = ScheduleVisualizer.plot_bin_utilization_heatmap(
            result=result,
            output_path=out_dir / "bin_utilization_heatmap.png",
        )
        pareto_path = ScheduleVisualizer.plot_pareto_frontier(
            base_config=config,
            output_path=out_dir / "pareto_frontier.png",
        )
        print(f"Generated Bin Utilization Heatmap: {heatmap_path}")
        print(f"Generated Pareto Frontier Plot:    {pareto_path}")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
