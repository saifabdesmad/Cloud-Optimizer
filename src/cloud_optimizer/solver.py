"""
Mixed-Integer Linear Programming (MILP) solver for heterogeneous cloud task scheduling.

Mathematical Formulation:
-------------------------
Sets:
  - Tasks: i in {1, ..., N}, with CPU request c_i, RAM request r_i,
    priority weight w_i in [1, 5], and execution deadline d_i.
  - Virtual Machines: j in {1, ..., M}, with CPU capacity C_j, RAM capacity R_j,
    hourly operating cost P_j, and instance reliability q_j in (0, 1].

Decision Variables:
  - x_{i,j} in {0, 1}: 1 if task i is assigned to VM j, 0 otherwise.
  - y_j in {0, 1}: 1 if VM j is turned ON, 0 otherwise.

Objective Function:
  min  sum_{j=1}^M P_j * y_j + lambda * sum_{i=1}^N w_i * (1 - sum_{j=1}^M x_{i,j})

Constraints:
  1. Single assignment: sum_{j=1}^M x_{i,j} <= 1   for all i in {1, ..., N}
  2. CPU capacity:      sum_{i=1}^N c_i * x_{i,j} <= C_j * y_j   for all j in {1, ..., M}
  3. RAM capacity:      sum_{i=1}^N r_i * x_{i,j} <= R_j * y_j   for all j in {1, ..., M}
  4. Total budget:      sum_{j=1}^M P_j * y_j <= B_max
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import pulp

from cloud_optimizer.models import (
    ClusterConfig,
    OptimizationResult,
    Task,
    VMAllocationDetail,
    VirtualMachine,
)


class CloudTaskScheduler:
    """
    MILP solver for scheduling containerized tasks onto heterogeneous VMs
    using PuLP and the COIN-OR CBC branch-and-cut solver.
    """

    def __init__(self, config: ClusterConfig) -> None:
        self.config: ClusterConfig = config
        self.tasks: List[Task] = config.tasks
        self.vms: List[VirtualMachine] = config.vms
        self.model: Optional[pulp.LpProblem] = None
        self.x_vars: Dict[Tuple[str, str], pulp.LpVariable] = {}
        self.y_vars: Dict[str, pulp.LpVariable] = {}
        self._last_result: Optional[OptimizationResult] = None

    def build_model(self) -> pulp.LpProblem:
        """
        Construct the PuLP MILP problem with decision variables x_{i,j}, y_j,
        objective function, and constraints (1)-(4).

        Returns:
            Configured pulp.LpProblem instance ready for solving.
        """
        prob = pulp.LpProblem("Cloud_Task_Scheduling_MILP", pulp.LpMinimize)

        # Decision Variables:
        # y_j in {0, 1} for j in {1, ..., M}
        self.y_vars = {
            vm.id: pulp.LpVariable(f"y_{vm.id}", lowBound=0, upBound=1, cat=pulp.LpBinary)
            for vm in self.vms
        }

        # x_{i,j} in {0, 1} for i in {1, ..., N}, j in {1, ..., M}
        self.x_vars = {
            (task.id, vm.id): pulp.LpVariable(
                f"x_{task.id}_{vm.id}", lowBound=0, upBound=1, cat=pulp.LpBinary
            )
            for task in self.tasks
            for vm in self.vms
        }

        # Objective Function:
        # min sum_{j=1}^M P_j * y_j + lambda * sum_{i=1}^N w_i * (1 - sum_{j=1}^M x_{i,j})
        operating_cost_expr = pulp.lpSum(
            vm.hourly_cost * self.y_vars[vm.id] for vm in self.vms
        )

        lam = self.config.lambda_penalty
        drop_penalty_expr = pulp.lpSum(
            lam
            * task.priority_weight
            * (1 - pulp.lpSum(self.x_vars[(task.id, vm.id)] for vm in self.vms))
            for task in self.tasks
        )

        prob += (
            operating_cost_expr + drop_penalty_expr,
            "Total_Operating_Cost_Plus_SLA_Penalty",
        )

        # Constraint 1: Each task is scheduled at most once
        # sum_{j=1}^M x_{i,j} <= 1   forall i in {1, ..., N}
        for task in self.tasks:
            prob += (
                pulp.lpSum(self.x_vars[(task.id, vm.id)] for vm in self.vms) <= 1,
                f"Task_Single_Assignment_{task.id}",
            )

        # Constraint 2: Machine CPU capacity
        # sum_{i=1}^N c_i * x_{i,j} <= C_j * y_j   forall j in {1, ..., M}
        for vm in self.vms:
            prob += (
                pulp.lpSum(
                    task.cpu_request * self.x_vars[(task.id, vm.id)]
                    for task in self.tasks
                )
                <= vm.cpu_capacity * self.y_vars[vm.id],
                f"VM_CPU_Capacity_{vm.id}",
            )

        # Constraint 3: Machine RAM capacity
        # sum_{i=1}^N r_i * x_{i,j} <= R_j * y_j   forall j in {1, ..., M}
        for vm in self.vms:
            prob += (
                pulp.lpSum(
                    task.ram_request * self.x_vars[(task.id, vm.id)]
                    for task in self.tasks
                )
                <= vm.ram_capacity * self.y_vars[vm.id],
                f"VM_RAM_Capacity_{vm.id}",
            )

        # Constraint 4: Total budget constraint
        # sum_{j=1}^M P_j * y_j <= B_max
        prob += (
            pulp.lpSum(vm.hourly_cost * self.y_vars[vm.id] for vm in self.vms)
            <= self.config.max_budget,
            "Cluster_Total_Budget_Ceiling",
        )

        self.model = prob
        return prob

    def _compute_ffd_incumbent(
        self,
    ) -> Tuple[Dict[str, Optional[str]], set[str], Dict[str, List[Task]]]:
        """
        Compute a fast greedy First-Fit Decreasing (FFD) incumbent solution
        and seed initial values on x_{i,j} and y_j.
        """
        max_cpu = max((v.cpu_capacity for v in self.vms), default=1.0)
        max_ram = max((v.ram_capacity for v in self.vms), default=1.0)

        sorted_tasks = sorted(
            self.tasks,
            key=lambda t: (
                t.priority_weight,
                (t.cpu_request / max_cpu) + (t.ram_request / max_ram),
            ),
            reverse=True,
        )
        sorted_vms = sorted(
            self.vms,
            key=lambda v: v.hourly_cost
            / ((v.cpu_capacity / max_cpu) + (v.ram_capacity / max_ram)),
        )

        rem_cpu = {v.id: v.cpu_capacity for v in self.vms}
        rem_ram = {v.id: v.ram_capacity for v in self.vms}
        active_vms: set[str] = set()
        current_spend = 0.0
        assigned_map: Dict[str, Optional[str]] = {t.id: None for t in self.tasks}
        vm_to_tasks: Dict[str, List[Task]] = {v.id: [] for v in self.vms}

        for task in sorted_tasks:
            placed = False
            for vm in sorted_vms:
                if vm.id in active_vms:
                    if (
                        rem_cpu[vm.id] >= task.cpu_request - 1e-9
                        and rem_ram[vm.id] >= task.ram_request - 1e-9
                    ):
                        rem_cpu[vm.id] -= task.cpu_request
                        rem_ram[vm.id] -= task.ram_request
                        assigned_map[task.id] = vm.id
                        vm_to_tasks[vm.id].append(task)
                        placed = True
                        break
            if placed:
                continue
            for vm in sorted_vms:
                if vm.id not in active_vms:
                    if (
                        rem_cpu[vm.id] >= task.cpu_request - 1e-9
                        and rem_ram[vm.id] >= task.ram_request - 1e-9
                        and current_spend + vm.hourly_cost
                        <= self.config.max_budget + 1e-9
                    ):
                        active_vms.add(vm.id)
                        current_spend += vm.hourly_cost
                        rem_cpu[vm.id] -= task.cpu_request
                        rem_ram[vm.id] -= task.ram_request
                        assigned_map[task.id] = vm.id
                        vm_to_tasks[vm.id].append(task)
                        break

        for vm in self.vms:
            self.y_vars[vm.id].setInitialValue(1.0 if vm.id in active_vms else 0.0)
        for task in self.tasks:
            chosen_vm = assigned_map[task.id]
            for vm in self.vms:
                self.x_vars[(task.id, vm.id)].setInitialValue(
                    1.0 if vm.id == chosen_vm else 0.0
                )

        return assigned_map, active_vms, vm_to_tasks

    def solve(self, msg: bool = False, warm_start: bool = True) -> OptimizationResult:
        """
        Solve the MILP model using CBC and extract the structured OptimizationResult.

        Args:
            msg: Whether to display CBC solver console output.
            warm_start: Seed the MIP solver with a greedy First-Fit Decreasing incumbent.

        Returns:
            Validated OptimizationResult containing assignments, costs, and utilization.
        """
        if self.model is None:
            self.build_model()

        assert self.model is not None

        incumbent_data = self._compute_ffd_incumbent() if warm_start else None

        solver = pulp.PULP_CBC_CMD(
            msg=msg,
            timeLimit=self.config.solver_time_limit,
            gapRel=self.config.mip_gap,
        )

        start_time = time.perf_counter()
        status_code = self.model.solve(solver)
        solve_time = time.perf_counter() - start_time

        status_str = pulp.LpStatus.get(status_code, "Unknown")

        # Extract binary decisions (rounding at 0.5 threshold to handle floating-point tolerances)
        active_vm_set = set()
        for vm in self.vms:
            val = pulp.value(self.y_vars[vm.id])
            if val is not None and round(float(val)) == 1:
                active_vm_set.add(vm.id)

        task_assignments: Dict[str, Optional[str]] = {t.id: None for t in self.tasks}
        vm_to_tasks: Dict[str, List[Task]] = {vm.id: [] for vm in self.vms}

        for task in self.tasks:
            for vm in self.vms:
                val = pulp.value(self.x_vars[(task.id, vm.id)])
                if val is not None and round(float(val)) == 1:
                    task_assignments[task.id] = vm.id
                    vm_to_tasks[vm.id].append(task)
                    active_vm_set.add(vm.id)
                    break

        cbc_result = build_optimization_result(
            strategy_name="MILP (CBC)",
            status=status_str,
            config=self.config,
            task_assignments=task_assignments,
            active_vm_ids=active_vm_set,
            vm_to_tasks=vm_to_tasks,
            solve_time_seconds=solve_time,
        )

        if incumbent_data is not None:
            inc_assign, inc_active, inc_vm_tasks = incumbent_data
            inc_result = build_optimization_result(
                strategy_name="MILP (CBC)",
                status="Feasible (Warm-Start Incumbent)",
                config=self.config,
                task_assignments=inc_assign,
                active_vm_ids=inc_active,
                vm_to_tasks=inc_vm_tasks,
                solve_time_seconds=solve_time,
            )
            if inc_result.objective_value < cbc_result.objective_value - 1e-6:
                for vm in self.vms:
                    self.y_vars[vm.id].varValue = (
                        1.0 if vm.id in inc_active else 0.0
                    )
                for task in self.tasks:
                    chosen_vm = inc_assign[task.id]
                    for vm in self.vms:
                        self.x_vars[(task.id, vm.id)].varValue = (
                            1.0 if vm.id == chosen_vm else 0.0
                        )
                self._last_result = inc_result
                return self._last_result

        self._last_result = cbc_result
        return self._last_result

    def get_allocation_summary(self) -> Dict[str, Any]:
        """
        Return a comprehensive dictionary summarizing the solved allocation,
        cost breakdown, SLA metrics, and constraint verification checks.

        Returns:
            Dictionary with summary statistics and per-VM allocation details.
        """
        if self._last_result is None:
            self.solve(msg=False)

        assert self._last_result is not None
        res = self._last_result

        total_tasks = len(self.tasks)
        scheduled_tasks = total_tasks - len(res.unassigned_tasks)

        return {
            "strategy": res.strategy_name,
            "status": res.status,
            "objective_value": round(res.objective_value, 4),
            "total_operating_cost_usd": round(res.total_operating_cost, 4),
            "max_budget_usd": round(self.config.max_budget, 4),
            "budget_utilization_pct": round(
                (res.total_operating_cost / self.config.max_budget * 100.0)
                if self.config.max_budget > 0
                else 0.0,
                2,
            ),
            "total_penalty_cost": round(res.total_penalty_cost, 4),
            "lambda_penalty": self.config.lambda_penalty,
            "tasks_total": total_tasks,
            "tasks_scheduled": scheduled_tasks,
            "tasks_dropped": len(res.unassigned_tasks),
            "sla_compliance_pct": round(res.sla_compliance_pct, 2),
            "vms_total": len(self.vms),
            "vms_active": len(res.active_vms),
            "cluster_cpu_utilization_pct": round(res.cluster_cpu_utilization_pct, 2),
            "cluster_ram_utilization_pct": round(res.cluster_ram_utilization_pct, 2),
            "mean_vm_cpu_utilization_pct": round(res.mean_cpu_utilization_pct, 2),
            "mean_vm_ram_utilization_pct": round(res.mean_ram_utilization_pct, 2),
            "solve_time_seconds": round(res.solve_time_seconds, 4),
            "active_vm_breakdown": [
                detail.model_dump()
                for detail in res.vm_details
                if detail.is_active == 1
            ],
        }


def build_optimization_result(
    strategy_name: str,
    status: str,
    config: ClusterConfig,
    task_assignments: Dict[str, Optional[str]],
    active_vm_ids: set[str],
    vm_to_tasks: Dict[str, List[Task]],
    solve_time_seconds: float,
) -> OptimizationResult:
    """
    Helper to compute standardized metrics and construct an OptimizationResult
    for both the MILP solver and heuristic baselines.
    """
    vm_details: List[VMAllocationDetail] = []
    active_vms_ordered: List[str] = []

    total_operating_cost = 0.0
    total_cpu_used = 0.0
    total_ram_used = 0.0
    total_active_cpu_cap = 0.0
    total_active_ram_cap = 0.0

    active_cpu_utils: List[float] = []
    active_ram_utils: List[float] = []

    for vm in config.vms:
        assigned = vm_to_tasks.get(vm.id, [])
        is_active = 1 if (vm.id in active_vm_ids or len(assigned) > 0) else 0
        cpu_used = sum(t.cpu_request for t in assigned)
        ram_used = sum(t.ram_request for t in assigned)

        cpu_util = (cpu_used / vm.cpu_capacity * 100.0) if is_active else 0.0
        ram_util = (ram_used / vm.ram_capacity * 100.0) if is_active else 0.0

        if is_active:
            active_vms_ordered.append(vm.id)
            total_operating_cost += vm.hourly_cost
            total_cpu_used += cpu_used
            total_ram_used += ram_used
            total_active_cpu_cap += vm.cpu_capacity
            total_active_ram_cap += vm.ram_capacity
            active_cpu_utils.append(cpu_util)
            active_ram_utils.append(ram_util)

        vm_details.append(
            VMAllocationDetail(
                vm_id=vm.id,
                instance_type=vm.instance_type,
                pricing_tier=vm.pricing_tier,
                is_active=is_active,
                assigned_tasks=[t.id for t in assigned],
                cpu_used=round(cpu_used, 4),
                cpu_capacity=vm.cpu_capacity,
                cpu_utilization_pct=round(cpu_util, 2),
                ram_used=round(ram_used, 4),
                ram_capacity=vm.ram_capacity,
                ram_utilization_pct=round(ram_util, 2),
                hourly_cost=vm.hourly_cost,
                reliability=vm.reliability,
            )
        )

    unassigned_tasks = [
        t.id for t in config.tasks if task_assignments.get(t.id) is None
    ]

    total_priority = sum(t.priority_weight for t in config.tasks)
    dropped_priority = sum(
        t.priority_weight
        for t in config.tasks
        if task_assignments.get(t.id) is None
    )

    total_penalty_cost = config.lambda_penalty * dropped_priority
    objective_value = total_operating_cost + total_penalty_cost

    mean_cpu_util = (
        sum(active_cpu_utils) / len(active_cpu_utils) if active_cpu_utils else 0.0
    )
    mean_ram_util = (
        sum(active_ram_utils) / len(active_ram_utils) if active_ram_utils else 0.0
    )

    cluster_cpu_util = (
        (total_cpu_used / total_active_cpu_cap * 100.0)
        if total_active_cpu_cap > 0
        else 0.0
    )
    cluster_ram_util = (
        (total_ram_used / total_active_ram_cap * 100.0)
        if total_active_ram_cap > 0
        else 0.0
    )

    sla_compliance_pct = (
        ((total_priority - dropped_priority) / total_priority * 100.0)
        if total_priority > 0
        else 100.0
    )

    return OptimizationResult(
        strategy_name=strategy_name,
        status=status,
        objective_value=round(objective_value, 6),
        total_operating_cost=round(total_operating_cost, 6),
        total_penalty_cost=round(total_penalty_cost, 6),
        unweighted_dropped_priority=float(dropped_priority),
        task_assignments=task_assignments,
        active_vms=active_vms_ordered,
        unassigned_tasks=unassigned_tasks,
        vm_details=vm_details,
        mean_cpu_utilization_pct=round(mean_cpu_util, 2),
        mean_ram_utilization_pct=round(mean_ram_util, 2),
        cluster_cpu_utilization_pct=round(cluster_cpu_util, 2),
        cluster_ram_utilization_pct=round(cluster_ram_util, 2),
        sla_compliance_pct=round(sla_compliance_pct, 2),
        solve_time_seconds=round(solve_time_seconds, 6),
    )
