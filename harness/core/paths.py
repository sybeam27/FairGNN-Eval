"""One place that knows where the method repositories live.

The external method repositories and the shared algorithm wrappers were moved
under `models/` (`models/<Repo>-main`, `models/algorithms`). Every adapter and
configuration interpreter resolves a repository through `repo()` instead of a
hard-coded `ROOT/<Repo>-main`, and importing this module puts the workspace
parent on `sys.path` so the `FairGate.models.algorithms.*` imports resolve from
any working directory. Nothing scientific depends on this file.
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WORKSPACE = os.path.dirname(ROOT)
MODELS = os.path.join(ROOT, "models")

if WORKSPACE not in sys.path:
    sys.path.insert(0, WORKSPACE)


def repo(name: str) -> str:
    """Path of a vendored method repository, e.g. repo('FairSIN-main')."""
    for base in (MODELS, ROOT):
        p = os.path.join(base, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"method repository {name!r} not found under {MODELS}")
