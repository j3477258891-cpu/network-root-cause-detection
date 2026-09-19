"""V59 wrapper around the shared coded-campaign result recorder."""

from pathlib import Path

import record_v58_result as recorder


ROOT = Path(__file__).resolve().parents[1]
recorder.OUT = ROOT / "experiments/v59_extended_coded_campaign"


if __name__ == "__main__":
    recorder.main()
