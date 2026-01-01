#!/usr/bin/env -S uv run python
"""Submit SLURM array job for multiple seeds with multiple trials."""

import argparse
import json
import subprocess
import time
from pathlib import Path
from datetime import datetime
import re


def get_job_progress(log_file):
    """Parse all [PROGRESS] lines from a SLURM log file."""
    if not log_file.exists():
        return []

    try:
        # Read entire file to get all progress markers
        with open(log_file, "r") as f:
            lines = f.readlines()

        # Extract all [PROGRESS] lines
        progress_lines = []
        for line in lines:
            if "[PROGRESS]" in line:
                match = re.search(r"\[PROGRESS\]\s*(.+)", line)
                if match:
                    progress_lines.append(match.group(1).strip())
        return progress_lines
    except Exception:
        return []


def main():
    parser = argparse.ArgumentParser(
        description="Submit SLURM array job for coder workflow on multiple seeds",
        epilog="Example: %(prog)s seed1.md seed2.md --trials 3 5 --steps '1 2' '1 2 3' --max-concurrent 10",
    )
    parser.add_argument("seeds", nargs="+", help="Paths to seed files")
    parser.add_argument(
        "--trials",
        nargs="+",
        type=int,
        help="Number of trials per seed (single value or one per seed)",
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        help="Steps to run per seed. Format: space-separated ints like '1 2' or single value for all seeds",
    )
    parser.add_argument(
        "--output-prefix", default="", help="Prefix for output folder names"
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        help="Maximum number of concurrent jobs (SLURM array limit)",
    )
    args = parser.parse_args()

    # Get absolute paths
    repo_root = Path(__file__).parent.parent.resolve()
    agents_dir = repo_root / "agents"

    # Parse trials
    if args.trials is None:
        trials_per_seed = [1] * len(args.seeds)
    elif len(args.trials) == 1:
        trials_per_seed = args.trials * len(args.seeds)
    elif len(args.trials) == len(args.seeds):
        trials_per_seed = args.trials
    else:
        parser.error(
            f"--trials must have 1 value (for all seeds) or {len(args.seeds)} values (one per seed)"
        )

    # Parse steps
    if args.steps is None:
        steps_per_seed = [[1, 2, 3]] * len(args.seeds)
    elif len(args.steps) == 1:
        # Single value applies to all seeds
        parsed_steps = [int(s) for s in args.steps[0].split()]
        steps_per_seed = [parsed_steps] * len(args.seeds)
    elif len(args.steps) == len(args.seeds):
        # One value per seed
        steps_per_seed = [[int(s) for s in step_str.split()] for step_str in args.steps]
    else:
        parser.error(
            f"--steps must have 1 value (for all seeds) or {len(args.seeds)} values (one per seed)"
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Build task list - each seed gets its own job_id for separate summaries
    tasks = []
    job_ids = {}  # Map job_id -> number of tasks for that seed

    for seed_path, num_trials, steps in zip(
        args.seeds, trials_per_seed, steps_per_seed
    ):
        seed_name = Path(seed_path).stem
        output_prefix = args.output_prefix
        if output_prefix:
            output_prefix = output_prefix + "_"
        job_id = f"{output_prefix}{seed_name}_{timestamp}"

        full_seed_path = (Path.cwd() / seed_path).resolve().as_posix()

        for trial in range(num_trials):
            tasks.append(
                {
                    "seed_path": full_seed_path,
                    "steps": steps,
                    "trial": trial,
                    "job_id": job_id,
                }
            )

        # Track how many tasks for this job_id
        job_ids[job_id] = num_trials

    logs_dir = agents_dir / "coder" / "logs"

    # Write config file as JSONL
    config_file = logs_dir / f".slurm_config_{timestamp}.jsonl"
    with open(config_file, "w") as f:
        for task in tasks:
            f.write(json.dumps(task) + "\n")

    # Submit array job
    num_tasks = len(tasks)
    array_spec = f"0-{num_tasks - 1}"
    if args.max_concurrent:
        array_spec += f"%{args.max_concurrent}"

    run_single_coder_script = agents_dir / "coder_runner.py"
    summarize_script = agents_dir / "summarize-results.py"

    sbatch_cmd = [
        "sbatch",
        "--job-name",
        f"causal_agents_{timestamp}",
        "--output",
        str(logs_dir / "%x_%A_%a.out"),
        "--error",
        str(logs_dir / "%x_%A_%a.err"),
        "--time",
        "2:00:00",  # 2hr
        "--gpus",
        "1",
        "--array",
        array_spec,
        "--chdir",
        str(repo_root),  # Set working directory to repo root
        # Unbuffered output to actually stream logs live
        "--wrap",
        f"PYTHONUNBUFFERED=1 uv run {run_single_coder_script} {config_file}",
    ]

    result = subprocess.run(sbatch_cmd, capture_output=True, text=True, check=True)
    slurm_job_id = result.stdout.strip().split()[-1]

    # Submit dependent summary jobs - one per seed (afterany = runs whether jobs succeed or fail)
    summary_job_ids = []
    summary_paths = []

    for job_id, expected_count in job_ids.items():
        summary_cmd = [
            "sbatch",
            "--job-name",
            f"summary_{job_id}",
            "--output",
            str(logs_dir / f"summary_{job_id}.out"),
            "--error",
            str(logs_dir / f"summary_{job_id}.err"),
            "--time",
            "10:00",  # 10 minutes should be plenty for summary
            "--dependency",
            f"afterany:{slurm_job_id}",
            "--chdir",
            str(repo_root),  # Set working directory to repo root
            "--wrap",
            f"uv run {summarize_script} {job_id} {expected_count}",
        ]

        result = subprocess.run(summary_cmd, capture_output=True, text=True, check=True)
        summary_job_id = result.stdout.strip().split()[-1]
        summary_job_ids.append(summary_job_id)

        summary_path = Path(
            f"/mnt/polished-lake/data/causal_agent_runs/{job_id}/summary.txt"
        )
        summary_paths.append(summary_path)

    print(f"Submitted SLURM array job: {slurm_job_id}")
    print(
        f"Submitted {len(summary_job_ids)} summary jobs: {', '.join(summary_job_ids)}"
    )
    print(f"Config file: {config_file}")
    print(f"Array range: {array_spec} ({num_tasks} tasks)")
    if args.max_concurrent:
        print(f"Max concurrent: {args.max_concurrent}")
    print(f"\n{'=' * 60}")
    for i, task in enumerate(tasks):
        print(
            f"Task {i}: {task['seed_path']} (steps: {task['steps']}) for trial {task['trial']} of job {task['job_id']}"
        )
    print(f"{'=' * 60}")

    # Wait for all jobs to complete
    print("\nWaiting for jobs to complete...")
    print("=" * 80)
    last_progress = {}  # Track progress per task to only show updates
    all_job_ids = f"{slurm_job_id},{','.join(summary_job_ids)}"
    while True:
        result = subprocess.run(
            ["squeue", "-j", all_job_ids, "-h"], capture_output=True, text=True
        )
        if not result.stdout.strip():
            break

        # Check progress for each task and print only NEW updates
        for i, task in enumerate(tasks):
            # Look for the log file for this task
            log_file = logs_dir / f"causal_agents_{timestamp}_{slurm_job_id}_{i}.out"
            progress_lines = get_job_progress(log_file)

            if not progress_lines:
                continue

            # Find where we left off
            last_printed = last_progress.get(i)
            if last_printed and last_printed in progress_lines:
                # Find index and print everything after it
                start_index = progress_lines.index(last_printed) + 1
                new_lines = progress_lines[start_index:]
            else:
                # First time or can't find last line - print everything
                new_lines = progress_lines

            # Print new progress lines
            if new_lines:
                seed_name = Path(task["seed_path"]).stem
                for line in new_lines:
                    time_str = datetime.now().strftime("%H:%M:%S")
                    print(f"  [{time_str}] [{seed_name} trial {task['trial']}] {line}")
                # Update to the last line we printed
                last_progress[i] = progress_lines[-1]

        time.sleep(10)

    # Print all summaries
    print("\n" + "=" * 80)
    print("ALL SUMMARIES")
    print("=" * 80)

    for summary_path in summary_paths:
        if summary_path.exists():
            print(summary_path.read_text())
            print()  # Extra newline between summaries
        else:
            print(f"Warning: Summary file not found at {summary_path}")
            print()

    # Ring terminal bell to notify completion
    print("\a")


if __name__ == "__main__":
    main()
