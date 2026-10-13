"""Run the test suites. One entry point for CI and for people.

    python run_tests.py                # everything the environment supports
    python run_tests.py --list         # what would run, and what is missing
    python run_tests.py test_folds     # just these

Suites are skipped, not failed, when their dependencies are absent -- test_folds
runs on a bare Python, while the model suites need torch. A skip is reported
loudly so a CI job that installed nothing cannot look like a green run.
"""

import argparse
import importlib.util
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

#: suite -> third-party modules it needs
SUITES = {
    "test_folds": (),
    "test_augment": ("numpy", "cv2"),
    "test_segmetrics": ("numpy", "scipy"),
    "test_analyze": ("numpy", "scipy"),
    "test_backbones": ("torch",),
    "test_conditioning": ("torch",),
    "test_training": ("torch", "cv2", "numpy"),
    "test_experiment": ("torch", "cv2", "numpy"),
}


def missing_requirements(names):
    return [n for n in names if importlib.util.find_spec(n) is None]


def run(suite):
    """Run one suite in a subprocess. Returns (status, passed, total)."""
    absent = missing_requirements(SUITES[suite])
    if absent:
        return "skip", 0, 0, f"needs {', '.join(absent)}"

    proc = subprocess.run([sys.executable, f"{suite}.py"], cwd=HERE,
                          capture_output=True, text=True)
    tail = (proc.stdout or "").strip().splitlines()
    summary = tail[-1] if tail else "(no output)"

    passed = total = 0
    if "/" in summary and "passed" in summary:
        try:
            counts = summary.split()[0]
            passed, total = (int(x) for x in counts.split("/"))
        except ValueError:
            pass

    if proc.returncode != 0:
        failures = [l.strip() for l in tail if l.strip().startswith(("FAIL", "ERROR"))]
        return "fail", passed, total, "; ".join(failures[:3]) or summary
    return "pass", passed, total, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("suites", nargs="*", default=None,
                        help="suite names to run (default: all)")
    parser.add_argument("--list", action="store_true",
                        help="show which suites this environment can run")
    args = parser.parse_args()

    wanted = args.suites or list(SUITES)
    unknown = [s for s in wanted if s not in SUITES]
    if unknown:
        raise SystemExit(f"Unknown suite(s): {', '.join(unknown)}. "
                         f"Available: {', '.join(SUITES)}")

    if args.list:
        for suite in wanted:
            absent = missing_requirements(SUITES[suite])
            state = f"missing {', '.join(absent)}" if absent else "ready"
            print(f"  {suite:<20}{state}")
        return

    width = max(len(s) for s in wanted)
    results, passed_total, test_total = [], 0, 0

    for suite in wanted:
        status, passed, total, detail = run(suite)
        results.append((status, suite, detail))
        passed_total += passed
        test_total += total
        mark = {"pass": "ok", "fail": "FAIL", "skip": "skip"}[status]
        print(f"  {suite:<{width}}  {mark:<5} {detail}")

    failed = [r for r in results if r[0] == "fail"]
    skipped = [r for r in results if r[0] == "skip"]

    print(f"\n{passed_total}/{test_total} tests passed across "
          f"{len(results) - len(skipped)} suite(s)")
    if skipped:
        print(f"SKIPPED {len(skipped)} suite(s): "
              f"{', '.join(s for _, s, _ in skipped)} -- install their "
              f"dependencies to run them.")
    if failed:
        print(f"FAILED {len(failed)} suite(s): {', '.join(s for _, s, _ in failed)}")

    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
