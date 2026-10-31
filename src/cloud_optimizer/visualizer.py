"""
Visualization module for cloud_optimizer.

Generates publication-quality Matplotlib figures:
1. Bin Utilization Heatmap: CPU and RAM utilization percentages across VMs.
2. Cost vs. Task Drop Tradeoff Pareto Frontier: Sweeps penalty factor lambda
   (and/or budget ceilings) to trace the Pareto-optimal frontier between hourly
   infrastructure expenditure ($) and unscheduled priority weight / task drops.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cloud_optimizer.models import ClusterConfig, OptimizationResult
from cloud_optimizer.solver import CloudTaskScheduler


class ScheduleVisualizer:
    """
    Generates diagnostic plots for MILP and heuristic scheduling solutions.
    """

    @staticmethod
    def plot_bin_utilization_heatmap(
        result: OptimizationResult,
        output_path: str | Path = "bin_utilization_heatmap.png",
        only_active: bool = False,
    ) -> Path:
        """
        Render a 2D heatmap of CPU (%) and RAM (%) utilization across virtual machines.

        Args:
            result: Solved OptimizationResult object.
            output_path: Destination image path.
            only_active: If True, filter heatmap to active VMs (y_j = 1) only.

        Returns:
            Resolved Path to the saved PNG plot.
        """
        out = Path(output_path).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)

        details = [
            d for d in result.vm_details if (not only_active or d.is_active == 1)
        ]
        if not details:
            details = result.vm_details

        vm_labels = [
            f"{d.vm_id}\n({d.instance_type}, {'ON' if d.is_active else 'OFF'})"
            for d in details
        ]
        cpu_utils = [d.cpu_utilization_pct for d in details]
        ram_utils = [d.ram_utilization_pct for d in details]

        data = np.array([cpu_utils, ram_utils], dtype=float)

        fig_width = max(8.0, min(22.0, len(details) * 0.85 + 2.5))
        fig, ax = plt.subplots(figsize=(fig_width, 4.5), dpi=150)

        im = ax.imshow(data, cmap="YlGnBu", aspect="auto", vmin=0.0, vmax=100.0)

        ax.set_yticks([0, 1])
        ax.set_yticklabels(["CPU Utilization (%)", "RAM Utilization (%)"], fontsize=10, fontweight="bold")
        ax.set_xticks(np.arange(len(vm_labels)))
        ax.set_xticklabels(vm_labels, rotation=40, ha="right", fontsize=8)

        # Annotate each cell with exact utilization percentage
        for row_idx in range(data.shape[0]):
            for col_idx in range(data.shape[1]):
                val = data[row_idx, col_idx]
                text_color = "white" if val > 60.0 else "#1f2937"
                ax.text(
                    col_idx,
                    row_idx,
                    f"{val:.1f}%",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=8,
                    fontweight="bold",
                )

        cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.03)
        cbar.set_label("Capacity Utilization (%)", fontsize=9)

        ax.set_title(
            f"VM Resource Bin Utilization Heatmap — {result.strategy_name} "
            f"(Cost: ${result.total_operating_cost:.3f}/hr | SLA: {result.sla_compliance_pct:.1f}%)",
            fontsize=11,
            fontweight="bold",
            pad=12,
        )

        fig.tight_layout()
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        return out

    @staticmethod
    def compute_pareto_frontier(
        base_config: ClusterConfig,
        lambda_values: Optional[Sequence[float]] = None,
    ) -> List[Tuple[float, OptimizationResult]]:
        """
        Sweep the penalty scaling parameter lambda across multiple orders of magnitude
        to trace the Cost vs. Task Drop Pareto frontier.

        Args:
            base_config: Base ClusterConfig problem instance.
            lambda_values: Sequence of lambda penalty multipliers to evaluate.

        Returns:
            List of (lambda_value, OptimizationResult) tuples.
        """
        if lambda_values is None:
            lambda_values = [0.005, 0.015, 0.03, 0.05, 0.08, 0.12, 0.20, 0.50, 1.50, 5.0]

        frontier_points: List[Tuple[float, OptimizationResult]] = []
        for lam in lambda_values:
            cfg = base_config.model_copy(
                update={
                    "lambda_penalty": float(lam),
                    "solver_time_limit": min(base_config.solver_time_limit, 10.0),
                }
            )
            res = CloudTaskScheduler(cfg).solve(msg=False)
            frontier_points.append((float(lam), res))

        return frontier_points

    @classmethod
    def plot_pareto_frontier(
        cls,
        base_config: ClusterConfig,
        output_path: str | Path = "pareto_frontier.png",
        lambda_values: Optional[Sequence[float]] = None,
    ) -> Path:
        """
        Plot the Cost ($) vs. Unassigned Task Penalty / Dropped Tasks Pareto Frontier.

        Args:
            base_config: Base ClusterConfig to evaluate across lambda values.
            output_path: Destination image path.
            lambda_values: Optional custom list of lambda penalty weights.

        Returns:
            Resolved Path to the saved PNG plot.
        """
        out = Path(output_path).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)

        points = cls.compute_pareto_frontier(base_config, lambda_values=lambda_values)

        costs = [res.total_operating_cost for _, res in points]
        dropped_counts = [len(res.unassigned_tasks) for _, res in points]
        dropped_weights = [res.unweighted_dropped_priority for _, res in points]
        lambdas = [lam for lam, _ in points]

        fig, ax1 = plt.subplots(figsize=(8.5, 5.2), dpi=150)

        # Sort by operating cost for clean Pareto line connection
        order = np.argsort(costs)
        sorted_costs = [costs[i] for i in order]
        sorted_weights = [dropped_weights[i] for i in order]
        sorted_counts = [dropped_counts[i] for i in order]
        sorted_lambdas = [lambdas[i] for i in order]

        ax1.plot(
            sorted_costs,
            sorted_weights,
            marker="o",
            linewidth=2.2,
            markersize=7,
            color="#0284c7",
            label="Unscheduled Priority Weight ($\\sum w_i (1 - \\sum x_{ij})$)",
        )

        for cost_val, w_val, lam_val, cnt_val in zip(
            sorted_costs, sorted_weights, sorted_lambdas, sorted_counts
        ):
            ax1.annotate(
                f"$\\lambda={lam_val:g}$\n({cnt_val} dropped)",
                xy=(cost_val, w_val),
                xytext=(6, 6),
                textcoords="offset points",
                fontsize=7.5,
                color="#0f172a",
                bbox=dict(boxstyle="round,pad=0.2", fc="#f8fafc", ec="#cbd5e1", alpha=0.85),
            )

        ax1.axvline(
            base_config.max_budget,
            color="#dc2626",
            linestyle="--",
            linewidth=1.5,
            label=f"Budget Ceiling $B_{{max}}$ (${base_config.max_budget:.2f}/hr)",
        )

        ax1.set_xlabel("Cluster Operating Cost ($\\sum P_j y_j$ in USD / hr)", fontsize=10, fontweight="bold")
        ax1.set_ylabel("Unscheduled SLA Priority Weight", fontsize=10, fontweight="bold", color="#0284c7")
        ax1.set_title(
            "Multi-Objective Pareto Frontier: Infrastructure Cost vs. Task Drop Penalty",
            fontsize=11,
            fontweight="bold",
            pad=12,
        )
        ax1.grid(True, linestyle=":", alpha=0.6)
        ax1.legend(loc="upper right", frameon=True, fontsize=9)

        fig.tight_layout()
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        return out
