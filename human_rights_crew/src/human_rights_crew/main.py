from __future__ import annotations

import argparse
import os
import sys
from dotenv import load_dotenv

load_dotenv()

from human_rights_crew.crew import HumanRightsCrew


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-q", "--question", required=True)
    args = ap.parse_args()

    if not os.getenv("MCP_URL"):
        print("[ERROR] MCP_URL is not set.", file=sys.stderr)
        sys.exit(1)

    c = HumanRightsCrew().crew()
    result = c.kickoff(inputs={"question": args.question})
    print(result)


if __name__ == "__main__":
    main()
