#!/usr/bin/env -S uv run python
"""Core module for running Claude Code coder workflows."""

import argparse
import json
import os
import subprocess
import sys
import traceback
import yaml
import anyio
from datetime import datetime, timezone
from pathlib import Path
from claude_agent_sdk import (
    ClaudeSDKClient,
    ClaudeAgentOptions,
    PermissionResultAllow,
    PermissionResultDeny,
    ToolPermissionContext,
)
from claude_agent_sdk.types import ResultMessage

from grader import grade_output, GradeOutput
from results_db import write_result, RunResult


def parse_models_from_seed(seed_path: str) -> list[str]:
    """Parse model names from the YAML section in a seed file.

    Returns:
        List of model names to pre-download
    """
    seed_content = Path(seed_path).read_text()

    # Extract YAML block
    start = seed_content.find("```yaml")
    if start == -1:
        return []

    end = seed_content.find("```", start + 7)
    if end == -1:
        return []

    yaml_content = seed_content[start + 7 : end].strip()
    data = yaml.safe_load(yaml_content)

    return data.get("models", []) if data else []


def preload_models(models: list[str], worktree_path: Path):
    """Pre-download models to HuggingFace cache. This also initializes the virtual environment.

    Args:
        models: List of HuggingFace model names to download
        worktree_path: Path to worktree (for running in correct environment)
    """
    if not models:
        return

    print(f"[PROGRESS] Preloading {len(models)} model(s)")

    try:
        subprocess.run(
            ["uv", "run", "neural/download_models.py"] + models,
            cwd=worktree_path,
            check=True,
        )
    except subprocess.CalledProcessError:
        print("[PROGRESS] Model download failed - agents will retry during Step 1")


def deny_file_access(file_path: str) -> bool:
    """Deny file access if it is in the references directory."""

    forbidden_paths = [
        "agents/coder/references/",
        "demos/",
        "tasks/IOI/",
        "tasks/general_addition/",
        "tasks/MCQA/",
        "tasks/entity_binding/",
        "tests/integration/",
        "tests/conftest.py",
        "tests/test_pyvene_core/conftest.py",
    ]

    for forbidden_path in forbidden_paths:
        if forbidden_path in file_path:
            return True

    return False


async def check_permission(
    tool_name: str, tool_input: dict, context: ToolPermissionContext
) -> PermissionResultAllow | PermissionResultDeny:
    allowed_tools = [
        "Bash",
        "Read",
        "Write",
        "Edit",
        "Glob",
        "Grep",
        "WebSearch",
        "WebFetch",
        "Task",
        "TodoWrite",
    ]

    if tool_name not in allowed_tools:
        return PermissionResultDeny(
            message=f"{tool_name} is not allowed. Please note this issue in issues.md."
        )

    reward_hacking_denial = PermissionResultDeny(
        message="Cannot access reference implementation files. No reward hacking allowed! Please note this issue in issues.md."
    )

    """Fine-grained permission checks (allowed_tools already filtered which tools are available)."""
    # Bash - block git push and reference file access
    if tool_name == "Bash":
        command = tool_input.get("command", "")
        if "git push" in command:
            return PermissionResultDeny(
                message="git push is not allowed. Please note this issue in issues.md."
            )
        if deny_file_access(command):
            return reward_hacking_denial

    # Read - block reference files to prevent reward hacking
    if tool_name == "Read":
        file_path = tool_input.get("file_path", "")
        if deny_file_access(file_path):
            return reward_hacking_denial

    # Glob - block reference files
    if tool_name == "Glob":
        path = tool_input.get("path", "")
        pattern = tool_input.get("pattern", "")
        if deny_file_access(path) or deny_file_access(pattern):
            return reward_hacking_denial

    # Grep - block reference files
    if tool_name == "Grep":
        path = tool_input.get("path", "")
        if path and deny_file_access(path):
            return reward_hacking_denial

    return PermissionResultAllow()


def make_prompt(step: int, path_to_seed: str) -> str:
    base = f"Please follow the instructions in agents/coder/instructions/GUIDELINES.md and STEP_{step}.md."
    if step != 3:
        base += f" PATH_TO_SEED is {path_to_seed}. Put all new files, folders, and results in agents/coder/outputs."
    return base


def validate_steps(steps: list[int]):
    if not steps or steps[0] != 1:
        raise ValueError("Steps must start at 1")
    if steps != list(range(1, len(steps) + 1)):
        raise ValueError("Steps must be contiguous (e.g., [1], [1,2], [1,2,3])")


def save_sessions(cwd: Path, session_ids: dict, outputs_dir: Path):
    cwd_str = (
        cwd.resolve().as_posix().replace("/", "-").replace(".", "-").replace("_", "-")
    )

    for step, session_id in session_ids.items():
        session_file = Path(
            f"~/.claude/projects/{cwd_str}/{session_id}.jsonl"
        ).expanduser()

        # replace summary with step name
        session_jsonl = session_file.read_text()
        last_line = session_jsonl.splitlines()[-1]
        last_uuid = last_line.split('"uuid":"')[1].split('"')[0]
        session_jsonl += (
            f'{{"type":"summary","summary":"Step {step}","leafUuid":"{last_uuid}"}}\n'
        )

        (outputs_dir / f"{session_id}.jsonl").write_text(session_jsonl)

    # write session_ids.json
    (outputs_dir / "session_ids.json").write_text(json.dumps(session_ids, indent=2))


async def run_coder(
    path_to_seed: str, job_id: str, steps: list[int] = None, trial: int = 0
):
    """Run coder workflow in a temporary git worktree.

    Args:
        path_to_seed: Path to the seed specification file
        output_folder: Output folder name
        steps: List of step numbers to run (must be contiguous starting from 1)
    """
    if steps is None:
        steps = [1, 2, 3]

    validate_steps(steps)

    started_at = datetime.now(timezone.utc)
    worktree_path = (
        Path("/mnt/polished-lake/data/causal_agent_runs")
        / "worktrees"
        / job_id
        / f"trial_{trial}"
    )
    worktree_path.parent.mkdir(parents=True, exist_ok=True)

    ref_name = None
    issues = ""
    grade_data: GradeOutput | None = None
    total_cost_usd = 0.0
    github_link = None

    # Capture stderr with timestamps for error reporting
    stderr_lines = []

    def capture_stderr(line: str):
        """Capture stderr output with timestamp for error reporting."""
        timestamp = datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3]
        stderr_lines.append(f"[{timestamp}] {line}")

    try:
        subprocess.run(
            ["git", "worktree", "add", str(worktree_path), "HEAD", "--detach"],
            check=True,
        )

        subprocess.run(["uv", "sync"], cwd=worktree_path, check=True)

        # Parse and preload models from seed before starting steps
        models = parse_models_from_seed(path_to_seed)
        preload_models(models, worktree_path)

        options = ClaudeAgentOptions(
            # The environment defaults to a buggy shared Claude Code; we want per-user Claude Code
            # model="sonnet[1m]", # TODO: change to sonnet[1m] if we get context anxiety again
            cli_path=Path.home() / ".claude" / "local" / "claude",
            system_prompt={"type": "preset", "preset": "claude_code"},
            permission_mode="default",
            can_use_tool=check_permission,
            cwd=str(worktree_path),
            env={
                # Allow Bash commands so that the agent can wait for long-running experiments
                "BASH_MAX_TIMEOUT_MS": "14400000",  # 4 hours maximum
                # Unset VIRTUAL_ENV to avoid uv warnings about mismatching virtual environments
                "VIRTUAL_ENV": "",
            },
            stderr=capture_stderr,  # Capture stderr for error reporting
        )

        local_seed_path = Path("agents") / "seeds" / Path(path_to_seed).name

        # Create client for stateful conversation with retry capability
        client = ClaudeSDKClient(options=options)
        await client.connect()

        session_ids = {}
        try:
            for step in steps:
                print(f"[PROGRESS] Starting Step {step} in steps {steps}")
                print("=" * 80)

                # Retry loop - client auto-resumes session on retry
                max_retries = 3
                retry_delay = 90  # seconds (high to account for transient API issues)

                for attempt in range(max_retries):
                    try:
                        # Send query (on first try) or resume prompt (on retry)
                        if attempt == 0:
                            await client.query(
                                make_prompt(step, local_seed_path),
                                session_id=f"step_{step}",
                            )
                        else:
                            # On retry, just send continuation - SDK auto-resumes the session
                            await client.query("Please continue")

                        # Receive response until ResultMessage
                        async for message in client.receive_response():
                            if isinstance(message, ResultMessage):
                                session_ids[step] = message.session_id
                                # Track cost per step
                                if message.total_cost_usd is not None:
                                    total_cost_usd += message.total_cost_usd

                        # Success - break retry loop
                        break

                    except Exception as e:
                        error_str = str(e).lower()
                        # Check if error is retryable (API errors, timeouts, etc)
                        is_retryable = any(
                            pattern in error_str
                            for pattern in [
                                "500",
                                "internal server error",
                                "api_error",
                                "rate_limit",
                                "timeout",
                                "overloaded",
                            ]
                        )

                        if is_retryable and attempt < max_retries - 1:
                            wait_time = retry_delay * (
                                2**attempt
                            )  # Exponential backoff
                            print(
                                f"[PROGRESS] ⚠ Retryable error on attempt {attempt + 1}/{max_retries}: {e}"
                            )
                            print(f"[PROGRESS]  Resuming in {wait_time}s...")
                            await anyio.sleep(wait_time)
                            # Client stays connected - next query() will resume automatically
                        else:
                            # Non-retryable or final attempt - re-raise
                            raise

                print(f"[PROGRESS] Completed Step {step} in steps {steps}")
                print("=" * 80)

                # Disconnect after each step to ensure next step gets a fresh session
                await client.disconnect()
                client = ClaudeSDKClient(options=options)
                await client.connect()

        finally:
            # Always disconnect client
            await client.disconnect()

        # Save session IDs for later interactive access
        outputs_dir = worktree_path / "agents" / "coder" / "outputs"
        outputs_dir.mkdir(parents=True, exist_ok=True)

        save_sessions(worktree_path, session_ids, outputs_dir)

        # Grade output if step 2
        if 2 in steps:
            grade_data = grade_output(path_to_seed, worktree_path)
            grade_file = worktree_path / "agents" / "coder" / "outputs" / "grade.json"
            grade_file.write_text(json.dumps(grade_data, indent=2))

        try:
            subprocess.run(
                ["git", "add", "-A"], cwd=worktree_path, check=True, text=True
            )
        except subprocess.CalledProcessError as e:
            error_msg = f"git add failed (exit code {e.returncode})"
            if e.stderr:
                error_msg += f"\nStderr: {e.stderr}"
            if e.stdout:
                error_msg += f"\nStdout: {e.stdout}"
            raise RuntimeError(error_msg) from e

        try:
            subprocess.run(
                ["git", "commit", "-n", "-m", f"Coder run: {job_id}/trial_{trial}"],
                cwd=worktree_path,
                check=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            error_msg = f"git commit failed (exit code {e.returncode})"
            if e.stderr:
                error_msg += f"\nStderr: {e.stderr}"
            if e.stdout:
                error_msg += f"\nStdout: {e.stdout}"
            raise RuntimeError(error_msg) from e

        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=worktree_path,
            check=True,
            capture_output=True,
            text=True,
        )
        commit_sha = result.stdout.strip()
        github_link = (
            f"https://github.com/goodfire-ai/causalab-internal/commit/{commit_sha}"
        )

        ref_name = f"refs/coder-runs/{job_id}/trial_{trial}"
        subprocess.run(["git", "update-ref", ref_name, commit_sha], check=True)

        # Push to origin with proper error handling
        try:
            subprocess.run(["git", "push", "origin", ref_name], check=True, text=True)
        except subprocess.CalledProcessError as e:
            error_msg = f"git push failed (exit code {e.returncode})"
            if e.stderr:
                error_msg += f"\nStderr: {e.stderr}"
            if e.stdout:
                error_msg += f"\nStdout: {e.stdout}"
            raise RuntimeError(error_msg) from e

        # Read issues.md if it exists
        issues_file = worktree_path / "agents" / "coder" / "outputs" / "issues.md"
        if issues_file.exists():
            issues = issues_file.read_text()

        print("[PROGRESS] Completed run")

    except Exception:
        # Append exception with full traceback to issues
        tb = traceback.format_exc()

        # Include captured stderr if available
        stderr_output = "\n".join(stderr_lines)
        timestamp = datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3]
        exception_msg = (
            f"\n\n---\n\n"
            f"[{timestamp}] Exception during run:\n\n"
            f"**Stderr output (with timestamps):**\n```\n{stderr_output}\n```\n\n"
            f"**Traceback:**\n```\n{tb}```"
        )

        issues += exception_msg
        raise

    finally:
        completed_at = datetime.now(timezone.utc)

        # Write to results database
        result = RunResult(
            job_id=job_id,
            trial=trial,
            seed_path=path_to_seed,
            steps=steps,
            github_link=github_link or "unknown",
            git_ref=ref_name or "unknown",
            started_at=started_at,
            completed_at=completed_at,
            grade=grade_data["score"] if grade_data else None,
            output=grade_data["output"] if grade_data else None,
            grading_details=grade_data["details"] if grade_data else None,
            issues=issues,
            total_cost_usd=total_cost_usd,
        )
        write_result(result)


async def main():
    """Main entry point for SLURM array jobs."""
    parser = argparse.ArgumentParser(
        description="Run single coder workflow from SLURM array job"
    )
    parser.add_argument("config_file", help="Path to JSONL config file")
    args = parser.parse_args()

    # Get task ID from SLURM
    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))

    # Read config file (JSONL format)
    with open(args.config_file) as f:
        lines = f.readlines()

    if task_id >= len(lines):
        print(f"Error: Task ID {task_id} out of range (only {len(lines)} tasks)")
        sys.exit(1)

    # Parse JSON line
    config = json.loads(lines[task_id])
    seed_path = config["seed_path"]
    steps = config["steps"]
    trial = config["trial"]
    job_id = config["job_id"]

    print(
        f"Task {task_id}: Running {seed_path} (steps: {steps}) for trial {trial} of job {job_id}"
    )
    await run_coder(seed_path, job_id, steps, trial)


if __name__ == "__main__":
    anyio.run(main)
