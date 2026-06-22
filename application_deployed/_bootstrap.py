from __future__ import annotations

import sys
from pathlib import Path


APP_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = APP_DIR.parent
PROJECT_ROOT = WORKSPACE_ROOT / "failure_memory_diffusion_project"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
