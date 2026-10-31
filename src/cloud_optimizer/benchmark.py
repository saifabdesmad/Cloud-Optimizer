"""
Comparative benchmark suite evaluating the MILP solver against standard
heuristic baselines across multiple problem scales (N = 20, 50, 200, 1000).

Heuristic Baselines:
1. First-Fit Decreasing (FFD):
   - Sorts tasks in descending order of priority weight and normalized resource demand.
   - Sorts VMs in ascending order of cost-per-unit-capacity.
   - Greedily assigns tasks to already-active VMs first; only activates a new VM if
     budget B_max permits and the penalty saved exceeds the VM activation cost.
2. Random / Round-Robin Assignment:
   - Shuffles VMs in a deterministic round-robin ring and assigns tasks sequentially
     whenever capacity and total cluster budget B_max allow.
"""

from __future__ import annotations

import random
import time
from typing import Dict, List, Optional, Sequence
from pydantic import BaseModel

from cloud_optimizer.generator import WorkloadGenerator
from cloud_optimizer.models import ClusterConfig, OptimizationResult, Task, VirtualMachine
from cloud_optimizer.solver import CloudTaskScheduler, build_optimization_result


class FirstFitDecreasingScheduler:
    """
    Greedy First-Fit Decreasing (FFD) heuristic for Multi-Dimensional Bin Packing
    under a cluster budget ceiling.
    """

    def __init__(self, config: ClusterConfig) -> None:
        self.config = config

    def solve(self) -> OptimizationResult:
        start_time = time.perf_counter()

        max_cpu = max((v.cpu_capacity for v in self.config.vms), default=1.0)
        max_ram = max((v.ram_capacity for v in self.config.vms), default=1.0)

        # Sort tasks: highest priority weight first, then largest normalized resource footprint
        sorted_tasks = sorted(
            self.config.tasks,
            key=lambda t: (
                t.priority_weight,
                (t.cpu_request / max_cpu) + (t.ram_request / max_ram),
            ),
            reverse=True,
        )

        # Sort VMs by cost efficiency: hourly_cost / (normalized CPU + RAM capacity)
        sorted_vms = sorted(
            self.config.vms,
            key=lambda v: v.hourly_cost / ((v.cpu_capacity / max_cpu) + (v.ram_capacity / max_ram)),
        )

        rem_cpu: Dict[str, float] = {v.id: v.cpu_capacity for v in self.config.vms}
        rem_ram: Dict[str, float] = {v.id: v.ram_capacity for v in self.config.vms}
        vm_cost: Dict[str, float] = {v.id: v.hourly_cost for v in self.config.vms}

        active_vms: set[str] = set()
        current_spend = 0.0
        task_assignments: Dict[str, Optional[str]] = {t.id: None for t in self.config.tasks}
        vm_to_tasks: Dict[str, List[Task]] = {v.id: [] for v in self.config.vms}

        for task in sorted_tasks:
            placed = False
            # 1. Try fitting into an already-active VM (zero marginal VM cost)
            for vm in sorted_vms:
                if vm.id in active_vms:
                    if rem_cpu[vm.id] >= task.cpu_request - 1e-9 and rem_ram[vm.id] >= task.ram_request - 1e-9:
                        rem_cpu[vm.id] -= task.cpu_request
                        rem_ram[vm.id] -= task.ram_request
                        task_assignments[task.id] = vm.id
                        vm_to_tasks[vm.id].append(task)
                        placed = True
                        break

            if placed:
                continue

            # 2. Otherwise, open the first cost-efficient inactive VM that fits task & budget
            for vm in sorted_vms:
                if vm.id not in active_vms:
                    if (
                        rem_cpu[vm.id] >= task.cpu_request - 1e-9
                        and rem_ram[vm.id] >= task.ram_request - 1e-9
                        and current_spend + vm_cost[vm.id] <= self.config.max_budget + 1e-9
                    ):
                        active_vms.add(vm.id)
                        current_spend += vm_cost[vm.id]
                        rem_cpu[vm.id] -= task.cpu_request
                        rem_ram[vm.id] -= task.ram_request
                        task_assignments[task.id] = vm.id
                        vm_to_tasks[vm.id].append(task)
                        break

        solve_time = time.perf_counter() - start_time
        return build_optimization_result(
            strategy_name="First-Fit Decreasing",
            status="Heuristic",
            config=self.config,
            task_assignments=task_assignments,
            active_vm_ids=active_vms,
            vm_to_tasks=vm_to_tasks,
            solve_time_seconds=solve_time,
        )


class RandomRoundRobinScheduler:
    """
    Baseline Round-Robin / Random scheduler that cycles through shuffled VMs
    and assigns tasks whenever capacity and budget constraints allow.
    """

    def __init__(self, config: ClusterConfig, seed: int = 42) -> None:
        self.config = config
        self.seed = seed

    def solve(self) -> OptimizationResult:
        start_time = time.perf_counter()
        rng = random.Random(self.seed)

        vms_ring: List[VirtualMachine] = list(self.config.vms)
        rng.shuffle(vms_ring)

        rem_cpu: Dict[str, float] = {v.id: v.cpu_capacity for v in self.config.vms}
        rem_ram: Dict[str, float] = {v.id: v.ram_capacity for v in self.config.vms}
        vm_cost: Dict[str, float] = {v.id: v.hourly_cost for v in self.config.vms}

        active_vms: set[str] = set()
        current_spend = 0.0
        task_assignments: Dict[str, Optional[str]] = {t.id: None for t in self.config.tasks}
        vm_to_tasks: Dict[str, List[Task]] = {v.id: [] for v in self.config.vms}

        num_vms = len(vms_ring)
        pointer = 0

        for task in self.config.tasks:
            checked = 0
            while checked < num_vms:
                vm = vms_ring[pointer]
                pointer = (pointer + 1) % num_vms
                checked += 1

                fits_resources = (
                    rem_cpu[vm.id] >= task.cpu_request - 1e-9
                    and rem_ram[vm.id] >= task.ram_request - 1e-9
                )
                if not fits_resources:
                    continue

                if vm.id in active_vms:
                    rem_cpu[vm.id] -= task.cpu_request
                    rem_ram[vm.id] -= task.ram_request
                    task_assignments[task.id] = vm.id
                    vm_to_tasks[vm.id].append(task)
                    break
                elif current_spend + vm_cost[vm.id] <= self.config.max_budget + 1e-9:
                    active_vms.add(vm.id)
                    current_spend += vm_cost[vm.id]
                    rem_cpu[vm.id] -= task.cpu_request
                    rem_ram[vm.id] -= task.ram_request
                    task_assignments[task.id] = vm.id
                    vm_to_tasks[vm.id].append(task)
                    break

        solve_time = time.perf_counter() - start_time
        return build_optimization_result(
            strategy_name="Random Round-Robin",
            status="Heuristic",
            config=self.config,
            task_assignments=task_assignments,
            active_vm_ids=active_vms,
            vm_to_tasks=vm_to_tasks,
            solve_time_seconds=solve_time,
        )


class BenchmarkRow(BaseModel):
    """Single comparative benchmark record for a given scale N and algorithm."""

    scale_n: int
    num_vms_m: int
    strategy: str
    objective_value: float
    operating_cost_usd: float
    cost_diff_vs_milp_usd: float
    objective_diff_vs_milp: float
    cluster_cpu_util_pct: float
    cluster_ram_util_pct: float
    sla_compliance_pct: float
    active_vms_count: int
    unassigned_tasks_count: int
    runtime_seconds: float


class BenchmarkRunner:
    """
    Runs multi-scale benchmarks across N = [20, 50, 200, 1000] comparing:
    - MILP (CBC)
    - First-Fit Decreasing (FFD)
    - Random Round-Robin
    """

    def __init__(
        self,
        scales: Sequence[int] = (20, 50, 200, 1000),
        seed: int = 42,
        solver_time_limit: float = 15.0,
        lambda_penalty: float = 5.0,
    ) -> None:
        self.scales = list(scales)
        self.seed = seed
        self.solver_time_limit = solver_time_limit
        self.lambda_penalty = lambda_penalty

    def run_single_scale(self, num_tasks: int) -> List[BenchmarkRow]:
        """Run MILP, FFD, and Random Round-Robin on an identical synthetic cluster of size N."""
        gen = WorkloadGenerator(seed=self.seed + num_tasks)
        config = gen.generate_cluster_config(
            num_tasks=num_tasks,
            budget_tightness=0.80,
            lambda_penalty=self.lambda_penalty,
            solver_time_limit=self.solver_time_limit,
        )

        milp_res = CloudTaskScheduler(config).solve(msg=False)
        ffd_res = FirstFitDecreasingScheduler(config).solve()
        rr_res = RandomRoundRobinScheduler(config, seed=self.seed).solve()

        rows: List[BenchmarkRow] = []
        for res in (milp_res, ffd_res, rr_res):
            rows.append(
                BenchmarkRow(
                    scale_n=num_tasks,
                    num_vms_m=len(config.vms),
                    strategy=res.strategy_name,
                    objective_value=round(res.objective_value, 4),
                    operating_cost_usd=round(res.total_operating_cost, 4),
                    cost_diff_vs_milp_usd=round(
                        res.total_operating_cost - milp_res.total_operating_cost, 4
                    ),
                    objective_diff_vs_milp=round(
                        res.objective_value - milp_res.objective_value, 4
                    ),
                    cluster_cpu_util_pct=res.cluster_cpu_utilization_pct,
                    cluster_ram_util_pct=res.cluster_ram_utilization_pct,
                    sla_compliance_pct=res.sla_compliance_pct,
                    active_vms_count=len(res.active_vms),
                    unassigned_tasks_count=len(res.unassigned_tasks),
                    runtime_seconds=res.solve_time_seconds,
                )
            )
        return rows

    def run_all(self) -> List[BenchmarkRow]:
        """Execute comparative benchmarks across all configured task scales."""
        all_rows: List[BenchmarkRow] = []
        for n in self.scales:
            all_rows.extend(self.run_single_scale(n))
        return all_rows

    @staticmethod
    def format_markdown_table(rows: Sequence[BenchmarkRow]) -> str:
        """Format benchmark comparison rows into a clean ASCII/Markdown table."""
        headers = [
            "N (Tasks)",
            "M (VMs)",
            "Strategy",
            "Obj Val",
            "VM Cost ($)",
            "Cost Diff ($)",
            "CPU Util (%)",
            "RAM Util (%)",
            "SLA (%)",
            "Dropped",
            "Runtime (s)",
        ]
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for r in rows:
            lines.append(
                f"| {r.scale_n:<9d} "
                f"| {r.num_vms_m:<7d} "
                f"| {r.strategy:<20s} "
                f"| {r.objective_value:>7.3f} "
                f"| {r.operating_cost_usd:>11.4f} "
                f"| {r.cost_diff_vs_milp_usd:>+13.4f} "
                f"| {r.cluster_cpu_util_pct:>12.2f} "
                f"| {r.cluster_ram_util_pct:>12.2f} "
                f"| {r.sla_compliance_pct:>7.2f} "
                f"| {r.unassigned_tasks_count:>7d} "
                f"| {r.runtime_seconds:>11.4f} |"
            )
        return "\n".join(lines)
