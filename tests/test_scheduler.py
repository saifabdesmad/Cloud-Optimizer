"""
Unit and integration test suite for cloud_optimizer.

Verifies:
1. Constraint 1: Each task is scheduled at most once (sum_j x_{i,j} <= 1).
2. Constraint 2 & 3: No machine CPU or RAM capacity overflow (sum_i c_i x_{i,j} <= C_j y_j, sum_i r_i x_{i,j} <= R_j y_j).
3. Constraint 4: Total budget ceiling enforced (sum_j P_j y_j <= B_max).
4. Binary variable integrity: x_{i,j} in {0, 1}, y_j in {0, 1}, and x_{i,j} == 1 => y_j == 1.
5. Objective function & priority trade-off behavior under tight budgets.
6. Heuristic baselines & visualization artifact generation.
"""

from __future__ import annotations

from pathlib import Path

import pulp
import pytest

from cloud_optimizer.benchmark import (
    BenchmarkRunner,
    FirstFitDecreasingScheduler,
    RandomRoundRobinScheduler,
)
from cloud_optimizer.generator import WorkloadGenerator
from cloud_optimizer.models import ClusterConfig, Task, VirtualMachine
from cloud_optimizer.solver import CloudTaskScheduler
from cloud_optimizer.visualizer import ScheduleVisualizer


@pytest.fixture
def synthetic_config() -> ClusterConfig:
    gen = WorkloadGenerator(seed=101)
    return gen.generate_cluster_config(
        num_tasks=25,
        num_vms=8,
        budget_tightness=0.65,
        lambda_penalty=6.0,
        solver_time_limit=15.0,
    )


def test_milp_binary_variable_integrity(synthetic_config: ClusterConfig) -> None:
    """Verify that all solved PuLP decision variables x_{i,j} and y_j are strictly binary {0, 1}."""
    scheduler = CloudTaskScheduler(synthetic_config)
    scheduler.build_model()
    result = scheduler.solve(msg=False)

    assert result.status in ("Optimal", "Feasible")

    # Verify y_j in {0, 1}
    for vm_id, y_var in scheduler.y_vars.items():
        val = pulp.value(y_var)
        assert val is not None
        assert abs(val - round(val)) <= 1e-5, f"y_{vm_id} is fractional: {val}"
        assert round(val) in (0, 1)

    # Verify x_{i,j} in {0, 1} and coupling x_{i,j} <= y_j
    for (task_id, vm_id), x_var in scheduler.x_vars.items():
        x_val = pulp.value(x_var)
        y_val = pulp.value(scheduler.y_vars[vm_id])
        assert x_val is not None and y_val is not None
        assert abs(x_val - round(x_val)) <= 1e-5, f"x_{task_id}_{vm_id} is fractional: {x_val}"
        assert round(x_val) in (0, 1)
        assert round(x_val) <= round(y_val), (
            f"Task {task_id} assigned to inactive VM {vm_id}!"
        )


def test_single_assignment_constraint(synthetic_config: ClusterConfig) -> None:
    """Constraint 1: Each task i is assigned to at most one VM (sum_j x_{i,j} <= 1)."""
    scheduler = CloudTaskScheduler(synthetic_config)
    scheduler.solve(msg=False)

    for task in synthetic_config.tasks:
        row_sum = sum(
            round(float(pulp.value(scheduler.x_vars[(task.id, vm.id)]) or 0.0))
            for vm in synthetic_config.vms
        )
        assert row_sum in (0, 1), f"Task {task.id} assigned {row_sum} times!"


def test_no_cpu_or_ram_capacity_overflow(synthetic_config: ClusterConfig) -> None:
    """Constraints 2 & 3: Assigned CPU and RAM on each VM never exceed C_j * y_j and R_j * y_j."""
    scheduler = CloudTaskScheduler(synthetic_config)
    result = scheduler.solve(msg=False)

    task_map = {t.id: t for t in synthetic_config.tasks}
    vm_map = {v.id: v for v in synthetic_config.vms}

    for detail in result.vm_details:
        vm = vm_map[detail.vm_id]
        y_j = detail.is_active
        cpu_sum = sum(task_map[t_id].cpu_request for t_id in detail.assigned_tasks)
        ram_sum = sum(task_map[t_id].ram_request for t_id in detail.assigned_tasks)

        assert cpu_sum <= vm.cpu_capacity * y_j + 1e-6, (
            f"CPU overflow on {vm.id}: {cpu_sum} > {vm.cpu_capacity * y_j}"
        )
        assert ram_sum <= vm.ram_capacity * y_j + 1e-6, (
            f"RAM overflow on {vm.id}: {ram_sum} > {vm.ram_capacity * y_j}"
        )
        assert detail.cpu_utilization_pct <= 100.0 + 1e-4
        assert detail.ram_utilization_pct <= 100.0 + 1e-4


def test_budget_ceiling_constraint() -> None:
    """Constraint 4: Total operating cost sum_j P_j * y_j never exceeds B_max, even under tight budgets."""
    tasks = [
        Task(id=f"t-{i}", cpu_request=2.0, ram_request=4.0, priority_weight=5, deadline=2.0)
        for i in range(10)
    ]
    vms = [
        VirtualMachine(
            id=f"vm-{j}",
            instance_type="c6i.xlarge",
            pricing_tier="on-demand",
            cpu_capacity=4.0,
            ram_capacity=8.0,
            hourly_cost=0.50,
            reliability=0.99,
        )
        for j in range(5)
    ]

    # Budget allows at most 2 VMs (2 * $0.50 = $1.00 <= $1.10)
    config = ClusterConfig(
        tasks=tasks,
        vms=vms,
        max_budget=1.10,
        lambda_penalty=100.0,
    )
    scheduler = CloudTaskScheduler(config)
    result = scheduler.solve(msg=False)

    assert result.total_operating_cost <= config.max_budget + 1e-6
    assert len(result.active_vms) == 2
    # Each VM holds at most 2 tasks (4 CPU / 2 CPU per task) -> 4 tasks scheduled, 6 dropped
    assert len(result.unassigned_tasks) == 6


def test_priority_selection_under_capacity_contention() -> None:
    """Verify MILP prioritizes high-weight tasks (w_i = 5) over low-weight tasks (w_i = 1)."""
    tasks = [
        Task(id="low-prio", cpu_request=4.0, ram_request=8.0, priority_weight=1, deadline=5.0),
        Task(id="high-prio", cpu_request=4.0, ram_request=8.0, priority_weight=5, deadline=1.0),
    ]
    vms = [
        VirtualMachine(
            id="single-vm",
            instance_type="m6i.xlarge",
            pricing_tier="on-demand",
            cpu_capacity=4.0,
            ram_capacity=8.0,
            hourly_cost=0.20,
            reliability=0.999,
        )
    ]
    config = ClusterConfig(tasks=tasks, vms=vms, max_budget=1.0, lambda_penalty=2.0)
    result = CloudTaskScheduler(config).solve(msg=False)

    assert result.task_assignments["high-prio"] == "single-vm"
    assert result.task_assignments["low-prio"] is None
    assert "low-prio" in result.unassigned_tasks


def test_zero_budget_forces_all_vms_off() -> None:
    """When B_max = 0 and all VMs have P_j > 0, no VMs can be turned on."""
    gen = WorkloadGenerator(seed=7)
    base_cfg = gen.generate_cluster_config(num_tasks=10, num_vms=4)
    zero_cfg = base_cfg.model_copy(update={"max_budget": 0.0})

    result = CloudTaskScheduler(zero_cfg).solve(msg=False)
    assert result.total_operating_cost == 0.0
    assert len(result.active_vms) == 0
    assert len(result.unassigned_tasks) == 10


def test_heuristics_respect_all_constraints(synthetic_config: ClusterConfig) -> None:
    """Verify First-Fit Decreasing and Random Round-Robin baselines satisfy all capacity & budget rules."""
    ffd_res = FirstFitDecreasingScheduler(synthetic_config).solve()
    rr_res = RandomRoundRobinScheduler(synthetic_config).solve()
    milp_res = CloudTaskScheduler(synthetic_config).solve(msg=False)

    for res in (ffd_res, rr_res):
        assert res.total_operating_cost <= synthetic_config.max_budget + 1e-6
        for detail in res.vm_details:
            assert detail.cpu_used <= detail.cpu_capacity * detail.is_active + 1e-6
            assert detail.ram_used <= detail.ram_capacity * detail.is_active + 1e-6

    # MILP objective must be optimal (less than or equal to both greedy heuristics)
    assert milp_res.objective_value <= ffd_res.objective_value + 1e-4
    assert milp_res.objective_value <= rr_res.objective_value + 1e-4


def test_visualizer_and_allocation_summary(
    synthetic_config: ClusterConfig, tmp_path: Path
) -> None:
    """Verify get_allocation_summary() and Matplotlib plot generation."""
    scheduler = CloudTaskScheduler(synthetic_config)
    result = scheduler.solve(msg=False)
    summary = scheduler.get_allocation_summary()

    assert summary["tasks_total"] == len(synthetic_config.tasks)
    assert summary["tasks_scheduled"] + summary["tasks_dropped"] == summary["tasks_total"]

    heatmap_file = ScheduleVisualizer.plot_bin_utilization_heatmap(
        result, output_path=tmp_path / "heatmap.png"
    )
    pareto_file = ScheduleVisualizer.plot_pareto_frontier(
        synthetic_config,
        output_path=tmp_path / "pareto.png",
        lambda_values=[0.05, 0.5, 5.0],
    )

    assert heatmap_file.exists() and heatmap_file.stat().st_size > 1000
    assert pareto_file.exists() and pareto_file.stat().st_size > 1000
