import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent      # 3_model
PROJECT_ROOT = ROOT.parent                  # Epitext_Project
sys.path.insert(0, str(PROJECT_ROOT))

import config  # 3_model/config.py
