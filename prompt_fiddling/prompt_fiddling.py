"""Finds prompts for an LLM-based PDDL planner that have different performance."""

from pathlib import Path
from prpl_llm_utils.models import OpenAIModel
from prpl_llm_utils.cache import SQLite3PretrainedLargeModelCache
import sys
import subprocess
import os
import tempfile
import argparse
import re
import json
from datetime import datetime


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


def parse_plan_from_response(response_text: str) -> list[str]:
    """Parse plan actions from LLM response, handling various formats including tags."""
    plan = []

    # Try to find plan between <plan> tags
    plan_match = re.search(r'<plan>(.*?)</plan>', response_text, re.DOTALL | re.IGNORECASE)
    if plan_match:
        text_to_parse = plan_match.group(1)
    else:
        text_to_parse = response_text

    # Extract all lines that look like PDDL actions
    for line in text_to_parse.strip().split('\n'):
        line = line.strip()
        if line.startswith('(') and line.endswith(')'):
            plan.append(line)

    return plan


def generate_alternative_prompts(llm: OpenAIModel, base_prompt: str, num_prompts: int) -> list[tuple[str, str]]:
    """Generate N alternative prompts with different styles and approaches."""
    meta_prompt = f"""You are an expert at prompt engineering for PDDL planning tasks. Given the base prompt below, generate {num_prompts} WILDLY DIFFERENT alternative prompts that could be used to ask an LLM to generate a PDDL plan.

BASE PROMPT:
{base_prompt}

Your alternative prompts should vary along these dimensions:
- Level of specificity (very detailed vs. very terse)
- Wording and style (formal, casual, instructive, etc.)
- Instructions for reasoning (chain-of-thought, step-by-step, direct answer, etc.)
- Output format (allow thinking/reasoning in the response, use XML tags like <plan></plan>, etc.)
- Examples and explanations (include examples, domain knowledge, constraints, etc.)

For prompts that allow reasoning, use <plan></plan> tags to wrap the final plan so it can be parsed.

Please output exactly {num_prompts} prompts in the following format:

<prompt id="1" name="Short descriptive name">
Full prompt text here...
</prompt>

<prompt id="2" name="Short descriptive name">
Full prompt text here...
</prompt>

...and so on.

Make the prompts as DIFFERENT as possible from each other. Be creative!"""

    print("Generating alternative prompts...")
    response = llm.query(meta_prompt)

    # Parse the prompts from the response
    prompts = []
    pattern = r'<prompt id="(\d+)" name="([^"]+)">(.*?)</prompt>'
    matches = re.findall(pattern, response.text, re.DOTALL)

    for prompt_id, name, content in matches:
        prompts.append((name.strip(), content.strip()))

    if len(prompts) < num_prompts:
        print(f"Warning: Only generated {len(prompts)} prompts instead of {num_prompts}")

    return prompts


def load_and_print_results(results_file: str) -> None:
    """Load results from a JSON file and print prompt -> score mapping."""
    with open(results_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Results from: {results_file}")
    print(f"Model: {data['metadata']['model']}")
    print(f"Domain: {data['metadata']['domain']}")
    print(f"Total problems: {data['metadata']['total_problems']}")
    print(f"Timestamp: {data['metadata']['timestamp']}")
    print(f"\n{'='*70}")
    print(f"PROMPT -> SCORE MAPPING")
    print(f"{'='*70}\n")

    for result in data['results']:
        print(f"Prompt: {result['name']}")
        print(f"  Score: {result['success_rate']:.1f}%")
        print(f"  Successes: {result['successes']}/{data['metadata']['total_problems']}")
        print()


def _main() -> None:
    parser = argparse.ArgumentParser(description="Generate and validate PDDL plans using an LLM")
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4.1",
        help="LLM model name to use (default: gpt-4.1)"
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="hiking",
        help="PDDL domain name to use (default: hiking)"
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=5,
        help="Number of alternative prompts to generate (default: 5)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for results (default: results_<domain>_<timestamp>.json)"
    )
    parser.add_argument(
        "--load",
        type=str,
        default=None,
        help="Load and display results from a JSON file"
    )
    parser.add_argument(
        "--prompts-dir",
        type=str,
        default=None,
        help="Directory containing prompt files (.txt) to evaluate instead of generating prompts"
    )
    args = parser.parse_args()

    # If load mode, just load and print results
    if args.load:
        load_and_print_results(args.load)
        return

    cache = SQLite3PretrainedLargeModelCache(Path(".llm_cache.db"))
    llm = OpenAIModel(args.model, cache)
    print(f"Using model: {args.model}")
    print(f"Using domain: {args.domain}\n")

    pddl_dir = Path(__file__).parent / "third_party" / "pddl"
    domain_dir = pddl_dir / args.domain
    domain_file = domain_dir / "domain.pddl"

    if not domain_file.exists():
        print(f"Error: Domain file {domain_file} does not exist")
        return

    with open(domain_file, "r", encoding="utf-8") as f:
        domain_text = f.read()

    # Collect all problem/task files from domain directory
    problem_files = sorted(list(domain_dir.glob("problem*.pddl")) + list(domain_dir.glob("task*.pddl")))

    if not problem_files:
        print(f"Error: No problem or task files found in {domain_dir}")
        return

    print(f"Found {len(problem_files)} problems in {domain_dir}\n")

    # Load or generate prompts
    if args.prompts_dir:
        # Load prompts from directory
        prompts_path = Path(args.prompts_dir)
        if not prompts_path.exists():
            print(f"Error: Prompts directory {args.prompts_dir} does not exist")
            return

        prompt_files = sorted(prompts_path.glob("*.txt"))
        if not prompt_files:
            print(f"Error: No .txt files found in {args.prompts_dir}")
            return

        all_prompts = []
        for prompt_file in prompt_files:
            with open(prompt_file, "r", encoding="utf-8") as f:
                prompt_text = f.read()
            prompt_name = prompt_file.stem
            all_prompts.append((prompt_name, prompt_text))

        print(f"Loaded {len(all_prompts)} prompts from {args.prompts_dir}\n")

        # Print all loaded prompts
        print(f"{'='*70}")
        print(f"ALL LOADED PROMPTS")
        print(f"{'='*70}\n")

        for idx, (name, prompt_text) in enumerate(all_prompts):
            print(f"PROMPT {idx}: {name}")
            print(f"{'-'*70}")
            print(prompt_text)
            print(f"\n")
    else:
        # Generate prompts using LLM
        base_prompt = """Given the following PDDL domain and problem, generate a valid plan to solve it.

DOMAIN:
{domain_text}

PROBLEM:
{problem_text}

Please provide a plan as a list of actions, one per line, in the format:
(action-name arg1 arg2 ...)

Only output the plan actions, nothing else."""

        alternative_prompts = generate_alternative_prompts(llm, base_prompt, args.num_prompts)
        print(f"Generated {len(alternative_prompts)} alternative prompts\n")

        # Print all generated prompts
        print(f"{'='*70}")
        print(f"ALL GENERATED PROMPTS")
        print(f"{'='*70}\n")

        print(f"PROMPT 0: Base Prompt")
        print(f"{'-'*70}")
        print(base_prompt)
        print(f"\n")

        for idx, (name, prompt_text) in enumerate(alternative_prompts, 1):
            print(f"PROMPT {idx}: {name}")
            print(f"{'-'*70}")
            print(prompt_text)
            print(f"\n")

        # Add the base prompt as prompt 0
        all_prompts = [("Base Prompt", base_prompt)] + alternative_prompts

    # Evaluate each prompt on all problems
    prompt_results = []

    for prompt_idx, (prompt_name, prompt_template) in enumerate(all_prompts):
        print(f"\n{'='*70}")
        print(f"EVALUATING PROMPT {prompt_idx}: {prompt_name}")
        print(f"{'='*70}\n")

        successes = 0
        failures = 0
        problem_results = []

        for problem_file in problem_files:
            with open(problem_file, "r", encoding="utf-8") as f:
                problem_text = f.read()

            # Format the prompt with domain and problem
            prompt = prompt_template.replace("{domain_text}", domain_text).replace("{problem_text}", problem_text)

            print(f"  {problem_file.name}...", end=" ", flush=True)

            response = llm.query(prompt)
            plan = parse_plan_from_response(response.text)

            is_valid, message = validate_plan(domain_file, problem_file, plan)

            if is_valid:
                successes += 1
                print("✓")
                problem_results.append((problem_file.name, True, message))
            else:
                failures += 1
                print(f"✗")
                problem_results.append((problem_file.name, False, message))

        success_rate = successes / len(problem_files) * 100 if problem_files else 0
        print(f"\n  Summary: {successes}/{len(problem_files)} correct ({success_rate:.1f}%)")

        prompt_results.append({
            "index": prompt_idx,
            "name": prompt_name,
            "prompt": prompt_template,
            "successes": successes,
            "failures": failures,
            "success_rate": success_rate,
            "problem_results": problem_results
        })

    # Sort prompts by success rate (descending)
    prompt_results.sort(key=lambda x: x["success_rate"], reverse=True)

    # Print final summary
    print(f"\n\n{'='*70}")
    print(f"FINAL RANKING (sorted by success rate)")
    print(f"{'='*70}\n")

    for rank, result in enumerate(prompt_results, 1):
        print(f"Rank {rank}: {result['name']} (Prompt {result['index']})")
        print(f"  Success rate: {result['success_rate']:.1f}% ({result['successes']}/{len(problem_files)})")
        print()

    # Print best prompt
    best_prompt = prompt_results[0]
    print(f"\n{'='*70}")
    print(f"BEST PERFORMING PROMPT: {best_prompt['name']}")
    print(f"{'='*70}")
    print(best_prompt['prompt'])
    print(f"\nSuccess rate: {best_prompt['success_rate']:.1f}%")
    print(f"{'='*70}")

    # Save results to JSON file
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"results_{args.domain}_{timestamp}.json"

    results_data = {
        "metadata": {
            "model": args.model,
            "domain": args.domain,
            "num_prompts": args.num_prompts,
            "total_problems": len(problem_files),
            "timestamp": datetime.now().isoformat()
        },
        "results": [
            {
                "index": r["index"],
                "name": r["name"],
                "prompt": r["prompt"],
                "successes": r["successes"],
                "failures": r["failures"],
                "success_rate": r["success_rate"],
                "problem_results": [
                    {"problem": name, "valid": valid, "message": msg}
                    for name, valid, msg in r["problem_results"]
                ]
            }
            for r in prompt_results
        ]
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Results saved to: {args.output}")
    print(f"Load with: python prompt_fiddling.py --load {args.output}")
    print(f"{'='*70}")


if __name__ == "__main__":
    _main()
