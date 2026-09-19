"""Read-only reuse of the frozen V152 training helpers, with isolated imports."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PRIOR = HERE.with_name('v152_error_repair_campaign')


def load_prior(name, dependencies=None):
    alias = '_v152_' + name
    if alias in sys.modules:
        return sys.modules[alias]
    saved = {k: sys.modules.get(k) for k in (dependencies or {})}
    try:
        sys.modules.update(dependencies or {})
        spec = importlib.util.spec_from_file_location(alias, PRIOR / (name + '.py'))
        module = importlib.util.module_from_spec(spec)
        sys.modules[alias] = module
        spec.loader.exec_module(module)
        return module
    finally:
        for k, value in saved.items():
            if value is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = value


c152 = load_prior('core')
t152 = load_prior('training', {'core': c152})
old152 = load_prior('campaign', {'core': c152})
