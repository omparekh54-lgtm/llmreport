"""Run the command-line tool with ``python -m llmreport ...`` (works even when llmreport.exe is not on PATH)."""

import sys

from .cli import main

sys.exit(main())
