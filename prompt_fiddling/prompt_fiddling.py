"""Finds prompts for an LLM-based PDDL planner that have different performance."""

from pathlib import Path
from prpl_llm_utils.models import OpenAIModel
from prpl_llm_utils.cache import SQLite3PretrainedLargeModelCache
import sys
import subprocess
import os
import tempfile


def validate_plan(domain_file: Path, problem_file: Path, plan: list[str]) -> tuple[bool, str]:
    """Use VAL to check if a plan solves a PDDL problem."""
    plan_str = ""
    for t, action in enumerate(plan):
        plan_str += f"{t}: {action}\n"
    plan_file = tempfile.NamedTemporaryFile(delete=False).name
    with open(plan_file, "w", encoding="utf-8") as f:
        f.write(plan_str)
    val_dir = Path(__file__).parent / "third_party" / "val"
    if sys.platform == "darwin":
        platform_dir = "darwin"
    else:
        assert sys.platform.startswith("linux")
        platform_dir = "linux64"
    val = val_dir / platform_dir / "Validate"
    cmd_str = f'"{val}" -v "{domain_file}" "{problem_file}" ' f'"{plan_file}"'
    output = subprocess.getoutput(cmd_str)
    os.remove(plan_file)
    if "Plan valid" in output:
        return True, "Plan succeeded."
    repair_phrase = "Plan Repair Advice:"
    if repair_phrase in output:
        msg = output[output.index(repair_phrase) + len(repair_phrase) :]
        msg, _ = msg.split("Failed plans:")
        msg = "NOTE: " + msg.strip()
    else:
        msg = "NOTE: The plan did not achieve the goal."
    return False, msg


def _main() -> None:
    cache = SQLite3PretrainedLargeModelCache(Path(".llm_cache.db"))
    llm = OpenAIModel("gpt-4o-mini", cache)
    # Example usage:
    # response = llm.query("What's a funny one liner?")


if __name__ == "__main__":
    _main()
