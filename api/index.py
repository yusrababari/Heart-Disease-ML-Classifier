"""Vercel serverless entry point.

Delegates all requests to the FastAPI application defined next to it in
api/main.py. Vercel rewrites `/api/*` paths to this function (see vercel.json),
which bundles everything in this directory (main.py, graphs.py, model assets).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from main import app  # noqa: E402

handler = app