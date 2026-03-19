import sys
import logging

# CRITICAL: Must be first — before any fastmcp/pydantic imports that might log
logging.basicConfig(
    stream=sys.stderr,
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

from obsidian_rag.cli import cli  # noqa: E402

if __name__ == "__main__":
    cli()
