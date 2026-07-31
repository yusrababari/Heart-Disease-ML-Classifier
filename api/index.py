"""Vercel serverless entry point.

Delegates all requests to the FastAPI application defined in main.py.
Vercel rewrites `/api/*` paths to this function (see vercel.json).
"""

from main import app

handler = app
