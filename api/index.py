"""Vercel Python entrypoint.

Vercel's @vercel/python builder detects the module-level ``app`` (an ASGI
FastAPI instance) and serves it. All routes (``/health``, ``/companies``,
``/ask`` …) are rewritten to this function by ``vercel.json``.

The deployed filesystem is read-only; ``retrieval.lexical_backend`` detects the
Vercel runtime and loads the prebuilt bm25s index bundled via
``vercel.json`` ``includeFiles`` — it never rebuilds under /var/task.
"""

from api.main import app  # noqa: F401  (re-exported for the Vercel builder)
