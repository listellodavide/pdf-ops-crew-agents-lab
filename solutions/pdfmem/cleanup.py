"""Delete a team's resident memory: unused index variants, or everything between groups.

    python -m pdfmem.cleanup --keep scalar        # after Lab 1 Block 1: drop the other variants
    python -m pdfmem.cleanup --all                # end of the day: indexes and Foundry memory store
    python -m pdfmem.cleanup --all --team g1-t3   # trainer, for any team
"""

from __future__ import annotations

import argparse
import os

from pdfmem import cloud, schema


def team_indexes() -> list[str]:
    prefix = f"pdfmem-{cloud.team()}-"
    return [n for n in cloud.index_client().list_index_names() if n.startswith(prefix)]


def main() -> int:
    p = argparse.ArgumentParser(description="Delete a team's indexes and memory store.")
    p.add_argument("--keep", choices=schema.VARIANTS, help="keep this chunks variant, delete the other two")
    p.add_argument("--all", action="store_true", help="delete every index and the Foundry memory store of the team")
    p.add_argument("--team", help="override PDFMEM_TEAM")
    args = p.parse_args()
    if args.team:
        os.environ["PDFMEM_TEAM"] = args.team
    if not args.keep and not args.all:
        p.error("use --keep <variant> or --all")
    for name in team_indexes():
        if args.all or (name.startswith(cloud.index_name("chunks")) and name != cloud.index_name("chunks", args.keep)):
            cloud.index_client().delete_index(name)
            print(f"[deleted] index {name}")
    if args.all:
        try:
            cloud.project().beta.memory_stores.delete(f"pdfmem-{cloud.team()}")
            print(f"[deleted] Foundry memory store pdfmem-{cloud.team()}")
        except Exception as e:                       # not created, or no permission
            print(f"[skip] memory store: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
