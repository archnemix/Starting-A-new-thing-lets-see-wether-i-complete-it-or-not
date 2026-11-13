"""`python -m archx` - the same CLI as the `archx` script.

It exists because `python -m <package>` is how a lot of people first try a tool, and because a
recipe that only works after an editable install onto PATH is a worse first experience than one
that works in the directory you just cloned. No logic here: the entry point does the work.
"""

from __future__ import annotations

import sys

from .cli.main import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
