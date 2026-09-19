"""Read-only reuse of the frozen V152 implementation, with isolated imports."""
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


c = load_prior('core')
t = load_prior('training', {'core': c})
old = load_prior('campaign', {'core': c})


def action_key(a):
    return (a['order_id'], a.get('remove_rid') or '', a.get('add_rid') or '')


def protect_snapshot():
    paths = [c.BASE]
    for directory in (PRIOR, HERE.with_name('v149_online_calibrated_addition_campaign'),
                      HERE.with_name('v150_adaptive_addition_campaign')):
        # Cache files and checkpoints are included; environments/junctions are not traversed.
        import os
        for parent, dirs, files in os.walk(directory, followlinks=False):
            dirs[:] = [d for d in dirs if d not in ('.venv', 'node_modules', '__pycache__')]
            paths.extend(Path(parent, name) for name in files)
    return {str(p): c.sha(p) for p in sorted(set(paths))}


def verify_snapshot(snapshot):
    changed = [p for p, h in snapshot.items() if not Path(p).is_file() or c.sha(p) != h]
    c.require(not changed, 'Protected prior artifacts changed: ' + str(changed[:5]))
    return len(snapshot)
