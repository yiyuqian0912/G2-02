"""Stable launcher name matching Phase I; currently exposes scaffold checks only."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from rind_phase2.experiments import main
if __name__ == '__main__':
    main()
