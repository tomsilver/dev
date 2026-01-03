"""Finds prompts for an LLM-based PDDL planner that have different performance."""

from pathlib import Path
from prpl_llm_utils.models import OpenAIModel
from prpl_llm_utils.cache import SQLite3PretrainedLargeModelCache
import sys
import subprocess
import os
import tempfile
import argparse


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
    parser = argparse.ArgumentParser(description="Generate and validate PDDL plans using an LLM")
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="LLM model name to use (default: gpt-4o-mini)"
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="hiking",
        help="PDDL domain name to use (default: hiking)"
    )
    parser.add_argument(
        "--seed",
        type=str,
        default="seed0",
        help="Seed folder to use (default: seed0)"
    )
    args = parser.parse_args()

    cache = SQLite3PretrainedLargeModelCache(Path(".llm_cache.db"))
    llm = OpenAIModel(args.model, cache)
    print(f"Using model: {args.model}")
    print(f"Using domain: {args.domain}")
    print(f"Using seed: {args.seed}\n")

    pg3_dir = Path(__file__).parent / "third_party" / "pg3"
    domain_file = pg3_dir / f"{args.domain}.pddl"
    seed_dir = pg3_dir / args.domain / args.seed

    with open(domain_file, "r", encoding="utf-8") as f:
        domain_text = f.read()

    problem_files = sorted(seed_dir.glob("problem*.pddl"))
    print(f"Found {len(problem_files)} problems in {seed_dir}\n")

    successes = 0
    failures = 0
    results = []

    for problem_file in problem_files:
        print(f"{'='*60}")
        print(f"Processing {problem_file.name}...")
        print(f"{'='*60}")

        with open(problem_file, "r", encoding="utf-8") as f:
            problem_text = f.read()

        prompt = f"""Given the following PDDL domain and problem, generate a valid plan to solve it.

DOMAIN:
{domain_text}

PROBLEM:
{problem_text}

Please provide a plan as a list of actions, one per line, in the format:
(action-name arg1 arg2 ...)

Only output the plan actions, nothing else."""

        print("Querying LLM for plan...")
        response = llm.query(prompt)

        plan = []
        for line in response.text.strip().split('\n'):
            line = line.strip()
            if line.startswith('(') and line.endswith(')'):
                plan.append(line)

        print(f"Extracted {len(plan)} actions from LLM response")

        is_valid, message = validate_plan(domain_file, problem_file, plan)

        if is_valid:
            successes += 1
            print(f"✓ Plan VALID")
            results.append((problem_file.name, True, message))
        else:
            failures += 1
            print(f"✗ Plan INVALID: {message}")
            results.append((problem_file.name, False, message))
        print()

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total problems: {len(problem_files)}")
    print(f"Successes: {successes}")
    print(f"Failures: {failures}")
    print(f"Success rate: {successes / len(problem_files) * 100:.1f}%")
    print(f"\nDetailed results:")
    for name, valid, msg in results:
        status = "✓" if valid else "✗"
        print(f"  {status} {name}: {'Valid' if valid else msg[:80]}")


if __name__ == "__main__":
    _main()
