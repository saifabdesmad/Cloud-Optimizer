"""
Pydantic data schemas for the cloud_optimizer package.

Defines formal data models for containerized tasks, virtual machine instances,
cluster configuration parameters, and MILP/heuristic optimization results.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


class Task(BaseModel):
    """
    Represents a containerized compute task i in {1, ..., N}.

    Attributes:
        id: Unique task identifier.
        cpu_request: CPU cores requested (c_i > 0).
        ram_request: RAM in GiB requested (r_i > 0).
        priority_weight: SLA priority weight w_i in [1, 5].
        deadline: Execution deadline timestamp/horizon in hours (d_i > 0).
        workload_type: Category of task workload (batch, burst, service).
    """

    id: str = Field(..., description="Unique identifier for the task")
    cpu_request: float = Field(..., gt=0.0, description="CPU request c_i (vCPUs)")
    ram_request: float = Field(..., gt=0.0, description="RAM request r_i (GiB)")
    priority_weight: int = Field(
        ..., ge=1, le=5, description="Priority weight w_i in [1, 5]"
    )
    deadline: float = Field(
        ..., gt=0.0, description="Execution deadline d_i (in hours)"
    )
    workload_type: Literal["batch", "burst", "microservice"] = Field(
        default="batch", description="Workload profile classification"
    )


class VirtualMachine(BaseModel):
    """
    Represents a heterogeneous Virtual Machine instance j in {1, ..., M}.

    Attributes:
        id: Unique machine identifier.
        instance_type: Cloud instance SKU name (e.g., m6i.xlarge, c6i.2xlarge).
        pricing_tier: Spot or on-demand instance classification.
        cpu_capacity: Total CPU core capacity (C_j > 0).
        ram_capacity: Total RAM capacity in GiB (R_j > 0).
        hourly_cost: Hourly operating cost in USD (P_j >= 0).
        reliability: Instance reliability probability q_j in (0, 1].
        max_uptime_window: Maximum guaranteed availability window in hours (used for deadline compatibility).
    """

    id: str = Field(..., description="Unique identifier for the VM")
    instance_type: str = Field(
        default="standard.medium", description="Cloud provider instance type name"
    )
    pricing_tier: Literal["on-demand", "spot"] = Field(
        default="on-demand", description="Pricing model (on-demand vs. spot)"
    )
    cpu_capacity: float = Field(..., gt=0.0, description="CPU capacity C_j (vCPUs)")
    ram_capacity: float = Field(..., gt=0.0, description="RAM capacity R_j (GiB)")
    hourly_cost: float = Field(
        ..., ge=0.0, description="Hourly operating cost P_j (USD/hr)"
    )
    reliability: float = Field(
        ..., gt=0.0, le=1.0, description="Instance reliability probability q_j in (0, 1]"
    )

    @field_validator("reliability")
    @classmethod
    def validate_reliability(cls, v: float) -> float:
        if v <= 0.0 or v > 1.0:
            raise ValueError("reliability q_j must be in the interval (0, 1].")
        return v


class ClusterConfig(BaseModel):
    """
    Configuration parameters governing the MILP optimization model.

    Attributes:
        tasks: List of N tasks to schedule.
        vms: List of M available candidate virtual machines.
        max_budget: Maximum total hourly cluster budget B_max (USD).
        lambda_penalty: Penalty scaling factor (lambda) balancing VM cost vs. task drop penalty.
        solver_time_limit: Maximum solver runtime in seconds.
        mip_gap: Relative MIP optimality gap tolerance.
        enforce_reliability_sla: Optional flag to block high-priority tasks on low-reliability spot VMs.
    """

    tasks: List[Task] = Field(..., min_length=1, description="Set of tasks {1, ..., N}")
    vms: List[VirtualMachine] = Field(
        ..., min_length=1, description="Set of virtual machines {1, ..., M}"
    )
    max_budget: float = Field(
        ..., ge=0.0, description="Total hourly budget ceiling B_max (USD)"
    )
    lambda_penalty: float = Field(
        default=10.0,
        ge=0.0,
        description="Penalty scaling factor lambda for unassigned tasks",
    )
    solver_time_limit: float = Field(
        default=30.0, gt=0.0, description="CBC solver timeout in seconds"
    )
    mip_gap: float = Field(
        default=0.01, ge=0.0, lt=1.0, description="Relative MIP gap tolerance"
    )

    @model_validator(mode="after")
    def validate_unique_ids(self) -> "ClusterConfig":
        task_ids = [t.id for t in self.tasks]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("Task IDs must be unique within ClusterConfig.")
        vm_ids = [v.id for v in self.vms]
        if len(vm_ids) != len(set(vm_ids)):
            raise ValueError("VirtualMachine IDs must be unique within ClusterConfig.")
        return self


class VMAllocationDetail(BaseModel):
    """Resource utilization breakdown for a single active or inactive VM."""

    vm_id: str
    instance_type: str
    pricing_tier: str
    is_active: int = Field(..., ge=0, le=1, description="Binary indicator y_j in {0, 1}")
    assigned_tasks: List[str] = Field(default_factory=list)
    cpu_used: float = Field(default=0.0, ge=0.0)
    cpu_capacity: float = Field(..., gt=0.0)
    cpu_utilization_pct: float = Field(default=0.0, ge=0.0)
    ram_used: float = Field(default=0.0, ge=0.0)
    ram_capacity: float = Field(..., gt=0.0)
    ram_utilization_pct: float = Field(default=0.0, ge=0.0)
    hourly_cost: float = Field(..., ge=0.0)
    reliability: float = Field(..., gt=0.0, le=1.0)


class OptimizationResult(BaseModel):
    """
    Structured output of a scheduling run (MILP or Heuristic baseline).

    Attributes:
        strategy_name: Name of the algorithm (e.g., 'MILP (CBC)', 'First-Fit Decreasing').
        status: Solver status string ('Optimal', 'Feasible', 'Infeasible', 'Heuristic').
        objective_value: Total objective function value: Cost + lambda * WeightedPenalty.
        total_operating_cost: Sum of P_j * y_j across all machines (USD).
        total_penalty_cost: lambda * sum_i w_i * (1 - sum_j x_ij).
        unweighted_dropped_priority: sum_i w_i * (1 - sum_j x_ij).
        task_assignments: Mapping of task_id -> assigned vm_id (or None if dropped).
        active_vms: List of vm_ids with y_j == 1.
        unassigned_tasks: List of task_ids where sum_j x_ij == 0.
        vm_details: Per-VM utilization metrics.
        mean_cpu_utilization_pct: Average CPU utilization across active VMs (0-100).
        mean_ram_utilization_pct: Average RAM utilization across active VMs (0-100).
        cluster_cpu_utilization_pct: Total CPU used / Total active CPU capacity * 100.
        cluster_ram_utilization_pct: Total RAM used / Total active RAM capacity * 100.
        sla_compliance_pct: Percentage of total priority-weighted workload scheduled.
        solve_time_seconds: Wall-clock execution time in seconds.
    """

    strategy_name: str
    status: str
    objective_value: float
    total_operating_cost: float
    total_penalty_cost: float
    unweighted_dropped_priority: float
    task_assignments: Dict[str, Optional[str]]
    active_vms: List[str]
    unassigned_tasks: List[str]
    vm_details: List[VMAllocationDetail]
    mean_cpu_utilization_pct: float
    mean_ram_utilization_pct: float
    cluster_cpu_utilization_pct: float
    cluster_ram_utilization_pct: float
    sla_compliance_pct: float
    solve_time_seconds: float
