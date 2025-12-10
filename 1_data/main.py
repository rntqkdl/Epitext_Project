import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent      # 1_data 폴더
PROJECT_ROOT = ROOT.parent                  # Epitext_Project
sys.path.insert(0, str(PROJECT_ROOT))

import config  # 1_data/config.py
