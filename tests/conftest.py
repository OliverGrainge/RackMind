"""Pytest configuration: ensure both dc_sim and agents packages are importable."""

import os
import sys

# Add src/ to sys.path so `import agents` and `import dc_sim` work in tests.
_src_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "src",
)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)
