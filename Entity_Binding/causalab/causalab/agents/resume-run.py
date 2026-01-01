#!/usr/bin/env -S uv run python
"""Resume a previous coder run by restoring its conversations and opening it in Claude Code.

This script:
1. Fetches all refs from the remote
2. Creates a git worktree for the specified ref
3. Copies the saved session files into the Claude Code directory
4. Launches `claude --resume` so you can select and continue the conversation
"""

import argparse
import subprocess
import sys
from pathlib import Path


def get_cwd_str(cwd: Path) -> str:
    """Construct the Claude Code directory name from a working directory path.

    Matches the logic in coder_runner.py:67
    """
    return (
        cwd.resolve().as_posix().replace("/", "-").replace(".", "-").replace("_", "-")
    )


def get_full_ref(ref: str) -> str:
    """Convert ref to full path if needed.

    Handles file paths from tab completion (e.g., refs/experiment_name/trial_0.json)
    by extracting the folder name and filename without extension.
    """
    # If it's a JSON file path from tab completion, extract the ref name
    if ref.endswith(".json"):
        path = Path(ref)
        # refs/experiment_name/trial_0.json -> experiment_name/trial_0
        ref = f"{path.parent.name}/{path.stem}"

    if ref.startswith("refs/"):
        return ref
    return f"refs/coder-runs/{ref}"


def main():
    parser = argparse.ArgumentParser(
        description="Resume a previous coder run by restoring its conversations",
        epilog=(
            "Example: %(prog)s seed_20251024_001903_trial_2\n"
            "Tab completion: %(prog)s refs/<TAB> to browse available runs"
        ),
    )
    parser.add_argument(
        "ref",
        help=(
            "Ref name or file path. "
            "Plain name: 'experiment_name/trial_0', "
            "Full ref: 'refs/coder-runs/experiment_name/trial_0', "
            "Tab complete: 'refs/experiment_name/trial_0.json'"
        ),
    )
    args = parser.parse_args()

    # Fetch refs
    subprocess.run(
        ["git", "fetch", "origin", "refs/coder-runs/*:refs/coder-runs/*"], check=True
    )

    # Get full ref path
    full_ref = get_full_ref(args.ref)

    # Create worktree name based on the ref
    ref_name = full_ref.replace("refs/coder-runs/", "").replace("/", "_")
    worktree_path = (
        Path("/mnt/polished-lake/data/causal_agent_runs")
        / "worktrees"
        / "resume"
        / ref_name
    )

    # Create or reuse worktree
    worktree_path.parent.mkdir(parents=True, exist_ok=True)

    if not worktree_path.exists():
        # Create new worktree
        subprocess.run(
            ["git", "worktree", "add", str(worktree_path), full_ref, "--detach"],
            check=True,
        )

    # Check for session files
    outputs_dir = worktree_path / "agents" / "coder" / "outputs"
    session_files = list(outputs_dir.glob("*.jsonl"))

    if not session_files:
        print(f"Error: No session files found in {outputs_dir}", file=sys.stderr)
        sys.exit(1)

    # Construct Claude Code directory path (same logic as coder_runner.py)
    cwd_str = get_cwd_str(worktree_path)
    claude_dir = Path(f"~/.claude/projects/{cwd_str}").expanduser()
    claude_dir.mkdir(parents=True, exist_ok=True)

    # Copy session files
    for session_file in session_files:
        if session_file.name != "session_ids.json":
            (claude_dir / session_file.name).write_text(session_file.read_text())

    print(f"Worktree: {worktree_path}")

    # Launch claude --resume
    subprocess.run(["claude", "--resume"], cwd=worktree_path)


if __name__ == "__main__":
    main()
