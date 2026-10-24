"""Automated test runner executing the complete evaluation suite with structured summary."""
import os
import subprocess
import sys
import time

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    print("=" * 70)
    print("GeoAI Hackathon 2026: Comprehensive Evaluation Suite Runner")
    print("=" * 70)
    t0 = time.time()

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/test_activities.py",
        "tests/test_query_parser.py",
        "tests/test_safety_guardrails.py",
        "tests/test_json_sanitizer.py",
        "tests/test_providers_mocked.py",
        "tests/test_grounding.py",
        "-v"
    ]
    
    print(f"Executing: {' '.join(cmd)}\n")
    proc = subprocess.run(cmd, cwd=ROOT_DIR)
    
    elapsed = round(time.time() - t0, 2)
    print("\n" + "=" * 70)
    if proc.returncode == 0:
        print(f"[SUCCESS] All test suites completed successfully in {elapsed}s!")
    else:
        print(f"[FAILURE] Test suite finished with exit code {proc.returncode} in {elapsed}s.")
    print("=" * 70)
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
