"""
Synthetic workload and heterogeneous cloud infrastructure generator.

Generates realistic cloud computing scenarios featuring:
- Batch jobs (high CPU/RAM, relaxed deadlines, medium priority)
- Burst traffic tasks (moderate resources, tight deadlines, high priority)
- Microservices (small footprint, strict SLA priority)
- Spot vs. On-Demand VM instances with realistic cost/reliability tradeoffs
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

from cloud_optimizer.models import ClusterConfig, Task, VirtualMachine


# Realistic cloud instance templates inspired by AWS EC2 families
INSTANCE_CATALOG: List[Dict[str, object]] = [
    # General Purpose - On-Demand
    {
        "instance_type": "m6i.large",
        "pricing_tier": "on-demand",
        "cpu": 2.0,
        "ram": 8.0,
        "base_cost": 0.096,
        "reliability_range": (0.995, 0.9999),
    },
    {
        "instance_type": "m6i.xlarge",
        "pricing_tier": "on-demand",
        "cpu": 4.0,
        "ram": 16.0,
        "base_cost": 0.192,
        "reliability_range": (0.995, 0.9999),
    },
    {
        "instance_type": "m6i.2xlarge",
        "pricing_tier": "on-demand",
        "cpu": 8.0,
        "ram": 32.0,
        "base_cost": 0.384,
        "reliability_range": (0.995, 0.9999),
    },
    # Compute Optimized - On-Demand
    {
        "instance_type": "c6i.xlarge",
        "pricing_tier": "on-demand",
        "cpu": 4.0,
        "ram": 8.0,
        "base_cost": 0.170,
        "reliability_range": (0.990, 0.999),
    },
    {
        "instance_type": "c6i.2xlarge",
        "pricing_tier": "on-demand",
        "cpu": 8.0,
        "ram": 16.0,
        "base_cost": 0.340,
        "reliability_range": (0.990, 0.999),
    },
    # Memory Optimized - On-Demand
    {
        "instance_type": "r6i.xlarge",
        "pricing_tier": "on-demand",
        "cpu": 4.0,
        "ram": 32.0,
        "base_cost": 0.252,
        "reliability_range": (0.995, 0.9999),
    },
    {
        "instance_type": "r6i.2xlarge",
        "pricing_tier": "on-demand",
        "cpu": 8.0,
        "ram": 64.0,
        "base_cost": 0.504,
        "reliability_range": (0.995, 0.9999),
    },
    # Spot Instances (~60-70% discount, lower reliability q_j)
    {
        "instance_type": "m6i.xlarge-spot",
        "pricing_tier": "spot",
        "cpu": 4.0,
        "ram": 16.0,
        "base_cost": 0.068,
        "reliability_range": (0.75, 0.92),
    },
    {
        "instance_type": "c6i.2xlarge-spot",
        "pricing_tier": "spot",
        "cpu": 8.0,
        "ram": 16.0,
        "base_cost": 0.115,
        "reliability_range": (0.70, 0.90),
    },
    {
        "instance_type": "r6i.2xlarge-spot",
        "pricing_tier": "spot",
        "cpu": 8.0,
        "ram": 64.0,
        "base_cost": 0.165,
        "reliability_range": (0.72, 0.91),
    },
    {
        "instance_type": "c6i.4xlarge-spot",
        "pricing_tier": "spot",
        "cpu": 16.0,
        "ram": 32.0,
        "base_cost": 0.230,
        "reliability_range": (0.68, 0.88),
    },
    {
        "instance_type": "m6i.4xlarge",
        "pricing_tier": "on-demand",
        "cpu": 16.0,
        "ram": 64.0,
        "base_cost": 0.768,
        "reliability_range": (0.995, 0.9999),
    },
]


class WorkloadGenerator:
    """
    Synthetic benchmark generator for containerized tasks and heterogeneous VM fleets.
    """

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.rng = random.Random(seed)

    def generate_tasks(
        self,
        num_tasks: int,
        batch_ratio: float = 0.45,
        burst_ratio: float = 0.35,
    ) -> List[Task]:
        """
        Generate N realistic containerized tasks across three workload archetypes:
        - 'batch': High CPU/RAM, longer deadlines, priority 1-3
        - 'burst': Medium CPU/RAM, short deadlines, priority 3-5
        - 'microservice': Low CPU/RAM, tight deadlines, priority 4-5

        Args:
            num_tasks: Number of tasks N.
            batch_ratio: Proportion of batch jobs.
            burst_ratio: Proportion of burst traffic tasks.

        Returns:
            List of validated Task objects.
        """
        if num_tasks <= 0:
            raise ValueError("num_tasks must be a positive integer.")

        tasks: List[Task] = []
        for i in range(1, num_tasks + 1):
            u = self.rng.random()
            if u < batch_ratio:
                w_type = "batch"
                cpu = round(self.rng.choice([1.0, 2.0, 4.0, 6.0]) * self.rng.uniform(0.8, 1.1), 2)
                ram = round(self.rng.choice([4.0, 8.0, 16.0, 24.0]) * self.rng.uniform(0.8, 1.1), 2)
                priority = self.rng.randint(1, 3)
                deadline = round(self.rng.uniform(4.0, 24.0), 2)
            elif u < batch_ratio + burst_ratio:
                w_type = "burst"
                cpu = round(self.rng.choice([0.5, 1.0, 2.0, 4.0]) * self.rng.uniform(0.9, 1.2), 2)
                ram = round(self.rng.choice([2.0, 4.0, 8.0, 12.0]) * self.rng.uniform(0.9, 1.2), 2)
                priority = self.rng.randint(3, 5)
                deadline = round(self.rng.uniform(0.5, 4.0), 2)
            else:
                w_type = "microservice"
                cpu = round(self.rng.choice([0.25, 0.5, 1.0, 2.0]), 2)
                ram = round(self.rng.choice([1.0, 2.0, 4.0, 8.0]), 2)
                priority = self.rng.randint(4, 5)
                deadline = round(self.rng.uniform(0.25, 2.0), 2)

            tasks.append(
                Task(
                    id=f"task-{i:04d}",
                    cpu_request=max(0.25, cpu),
                    ram_request=max(0.5, ram),
                    priority_weight=priority,
                    deadline=deadline,
                    workload_type=w_type,  # type: ignore[arg-type]
                )
            )
        return tasks

    def generate_vms(
        self,
        num_vms: int,
        spot_fraction: float = 0.4,
    ) -> List[VirtualMachine]:
        """
        Generate M heterogeneous virtual machines combining Spot and On-Demand pricing.

        Args:
            num_vms: Number of VMs M.
            spot_fraction: Target fraction of spot instances in the pool.

        Returns:
            List of validated VirtualMachine objects.
        """
        if num_vms <= 0:
            raise ValueError("num_vms must be a positive integer.")

        spot_templates = [t for t in INSTANCE_CATALOG if t["pricing_tier"] == "spot"]
        ondemand_templates = [
            t for t in INSTANCE_CATALOG if t["pricing_tier"] == "on-demand"
        ]

        vms: List[VirtualMachine] = []
        for j in range(1, num_vms + 1):
            is_spot = self.rng.random() < spot_fraction
            pool = spot_templates if is_spot else ondemand_templates
            template = self.rng.choice(pool)

            base_cost = float(template["base_cost"])  # type: ignore[arg-type]
            # Add slight regional/market price jitter (+/- 8%)
            jitter = self.rng.uniform(0.92, 1.08)
            hourly_cost = round(base_cost * jitter, 4)

            rel_min, rel_max = template["reliability_range"]  # type: ignore[misc]
            reliability = round(self.rng.uniform(float(rel_min), float(rel_max)), 4)

            vms.append(
                VirtualMachine(
                    id=f"vm-{j:03d}",
                    instance_type=str(template["instance_type"]),
                    pricing_tier=str(template["pricing_tier"]),  # type: ignore[arg-type]
                    cpu_capacity=float(template["cpu"]),  # type: ignore[arg-type]
                    ram_capacity=float(template["ram"]),  # type: ignore[arg-type]
                    hourly_cost=hourly_cost,
                    reliability=reliability,
                )
            )
        return vms

    def generate_cluster_config(
        self,
        num_tasks: int = 50,
        num_vms: Optional[int] = None,
        budget_tightness: float = 0.75,
        lambda_penalty: float = 5.0,
        solver_time_limit: float = 30.0,
    ) -> ClusterConfig:
        """
        Create a complete ClusterConfig sized appropriately for the task demand.

        Args:
            num_tasks: Number of tasks N.
            num_vms: Number of candidate VMs M. If None, sized so aggregate cluster
                     capacity is roughly 1.35x total task demand.
            budget_tightness: Fraction of total fleet cost allowed as B_max.
            lambda_penalty: Penalty weight lambda for dropped tasks.
            solver_time_limit: Solver time limit in seconds.

        Returns:
            Validated ClusterConfig instance.
        """
        tasks = self.generate_tasks(num_tasks=num_tasks)
        total_cpu_demand = sum(t.cpu_request for t in tasks)
        total_ram_demand = sum(t.ram_request for t in tasks)

        if num_vms is None:
            # Average VM has ~7 CPU and ~26 GiB RAM; provide ~1.35x capacity pool
            est_vms_cpu = int((total_cpu_demand * 1.35) / 6.5) + 2
            est_vms_ram = int((total_ram_demand * 1.35) / 22.0) + 2
            num_vms = max(5, max(est_vms_cpu, est_vms_ram))

        vms = self.generate_vms(num_vms=num_vms)

        # Ensure aggregate capacity is at least 1.15x demand so problem has rich packing choices
        while (
            sum(v.cpu_capacity for v in vms) < total_cpu_demand * 1.15
            or sum(v.ram_capacity for v in vms) < total_ram_demand * 1.15
        ):
            extra_idx = len(vms) + 1
            extra_vm = self.generate_vms(num_vms=1)[0]
            extra_vm.id = f"vm-{extra_idx:03d}"
            vms.append(extra_vm)

        total_fleet_cost = sum(v.hourly_cost for v in vms)
        max_budget = round(max(0.5, total_fleet_cost * budget_tightness), 4)

        return ClusterConfig(
            tasks=tasks,
            vms=vms,
            max_budget=max_budget,
            lambda_penalty=lambda_penalty,
            solver_time_limit=solver_time_limit,
        )
