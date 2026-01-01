"""Results database for tracking evaluation runs."""

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class RunResult:
    """Result data from a single evaluation run."""

    job_id: str
    trial: int
    seed_path: str
    steps: list[int]
    github_link: str
    git_ref: str
    started_at: datetime
    completed_at: datetime
    grade: float | None = None
    grading_details: dict[str, dict[str, Any]] | None = None
    output: dict[str, Any] | None = None
    issues: str | None = None
    total_cost_usd: float | None = None


def write_result(run_result: RunResult) -> Path:
    """Write a run result to the central results database.

    Args:
        run_result: RunResult dataclass containing all result data

    Returns:
        Path to the written result file

    Raises:
        OSError: If directory creation or file writing fails
    """
    # Get SLURM job info from environment
    slurm_job_id = os.environ.get("SLURM_ARRAY_JOB_ID")

    # Convert to int if available
    if slurm_job_id:
        slurm_job_id = int(slurm_job_id)

    # Create result object
    result = {
        "job_id": run_result.job_id,
        "trial": run_result.trial,
        "slurm_job_id": slurm_job_id,
        "seed_path": run_result.seed_path,
        "steps": run_result.steps,
        "github_link": run_result.github_link,
        "git_ref": run_result.git_ref,
        "started_at": run_result.started_at.isoformat(),
        "completed_at": run_result.completed_at.isoformat(),
        "grading_details": run_result.grading_details,
        "issues": run_result.issues,
        "output": run_result.output,
        "grade": run_result.grade,
        "total_cost_usd": run_result.total_cost_usd,
    }

    # Determine output path
    db_root = Path("/mnt/polished-lake/data/causal_agent_runs/")

    job_dir = db_root / run_result.job_id

    job_dir.mkdir(parents=True, exist_ok=True)

    # Write result
    result_file = job_dir / f"trial_{run_result.trial}.json"
    with open(result_file, "w") as f:
        json.dump(result, f, indent=2)

    return result_file
