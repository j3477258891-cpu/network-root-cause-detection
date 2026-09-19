"""Read-only adapters to the frozen V158 evidence and CSV implementation."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
PREVIOUS = EXP / 'v158_eight_attempt_campaign'
sys.path.append(str(PREVIOUS))
from common import (module, read, write, sha, digest, now, lock, require,
                    csv_nodes, apply_actions, validate_file, infer, verify_sources,
                    TZ, datetime)
from evidence import Equations


def previous():
    return module('_v159_previous', PREVIOUS / 'campaign.py').Campaign(PREVIOUS)
