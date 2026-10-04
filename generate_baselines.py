#!/usr/bin/env python
"""Regenerate the pytest-mpl baseline images.

Each test module keeps its baselines in its own directory
(tests/soccer/baseline/<module>/), matching the baseline_dir set by
helpers.mpl_kwargs. pytest-mpl's --mpl-generate-path is a single flat
directory, so this script runs pytest once per test module.

Usage: uv run python generate_baselines.py [module ...]
e.g.   uv run python generate_baselines.py heatmap sonar
       (no arguments regenerates every module)
"""
import subprocess
import sys
from pathlib import Path

TEST_DIR = Path(__file__).parent / 'tests' / 'soccer'


def main():
    only = set(sys.argv[1:])
    for test_file in sorted(TEST_DIR.glob('test_*.py')):
        group = test_file.stem.removeprefix('test_')
        if only and group not in only:
            continue
        baseline_dir = TEST_DIR / 'baseline' / group
        print(f'== {test_file.name} -> {baseline_dir.relative_to(TEST_DIR.parent.parent)}')
        result = subprocess.run([sys.executable, '-m', 'pytest', str(test_file),
                                 '-q', f'--mpl-generate-path={baseline_dir}'])
        if result.returncode:
            sys.exit(result.returncode)


if __name__ == '__main__':
    main()
