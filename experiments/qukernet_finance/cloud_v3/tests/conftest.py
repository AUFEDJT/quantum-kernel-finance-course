"""Make the documented course-root test command resolve the sibling modules."""
from pathlib import Path
import sys

MODULE_ROOT = Path(__file__).resolve().parents[2]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))
