#!/usr/bin/env python3
"""SMOKE: exercise TimeWarp's non-Docker auto-provisioning end-to-end.

Runs the real L1+L2 path the benchmark uses in auto mode:
  1. ensure_provisioned() — clone the upstream repo + run its idempotent setup.sh
     (conda env + Google-Drive/HuggingFace data) if not already set up,
  2. start_servers(1)     — launch wiki/news/webshop Flask apps and wait until healthy,
  3. assert all three URLs are reachable, then stop() them.

This is the plumbing unit tests can't reach (they mock subprocess/socket). It needs conda
and several GB of downloads on the first run; SKIPs cleanly when conda is absent.

Run under the cube's venv:
    cubes/timewarp/.venv/bin/python cubes/timewarp/scripts/smoke/timewarp_autoprovision.py

Prints SMOKE OK|FAIL|SKIP: timewarp_autoprovision  (exit 0|1|2).
"""

from __future__ import annotations

import shutil
import sys

from timewarp_cube import provisioning

NAME = "timewarp_autoprovision"


def main() -> int:
    if shutil.which("conda") is None:
        print(f"SMOKE SKIP: {NAME} (conda not found — auto mode needs the timewarp conda env)")
        return 2

    servers = None
    try:
        checkout = provisioning.ensure_provisioned()
        print(f"[smoke] provisioned at {checkout}")
        servers = provisioning.start_servers(checkout, ui_version=1)
        for site, url in servers.urls.items():
            if not provisioning.is_reachable(url):
                print(f"SMOKE FAIL: {NAME} ({site} not reachable at {url})")
                return 1
            print(f"[smoke] {site} reachable at {url}")
    except Exception as exc:  # noqa: BLE001 — smoke reports any failure as FAIL
        print(f"SMOKE FAIL: {NAME} ({type(exc).__name__}: {exc})")
        return 1
    finally:
        if servers is not None:
            servers.stop()

    print(f"SMOKE OK: {NAME}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
