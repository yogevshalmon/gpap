#!/usr/bin/env python3
"""Back-compat shim: the benchmark driver moved INTO the package (gpap/benchmark.py) so the
installed `gpap bench` entry point works from any directory — `scripts/` is repo-root-relative
and is not importable from an installed gpap. This wrapper keeps the documented
`PYTHONPATH=. python scripts/sample_benchmark.py ...` invocation (and the
`scripts.sample_benchmark.cmsgen_sample` import that eval/ uses) working unchanged.
"""
import sys

sys.path.insert(0, ".")
from gpap.benchmark import CIRCUIT_EXTS, cmsgen_sample, main  # noqa: F401,E402

if __name__ == "__main__":
    main()
