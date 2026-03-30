from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from grid_benchmark_common import CAMPAIGN_SLUG, REPO_ROOT  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch compact benchmark artifacts from the remote server.")
    parser.add_argument("--host", default="root@region-41.seetacloud.com")
    parser.add_argument("--port", type=int, default=44817)
    parser.add_argument("--remote-repo", default="~/ruivi")
    args = parser.parse_args()

    if shutil.which("ssh") is None:
        raise RuntimeError("ssh command not found on PATH")

    remote_cmd = f"""
set -e
cd {args.remote_repo}
TMP_LIST="$(mktemp)"
cleanup() {{
  rm -f "$TMP_LIST"
}}
trap cleanup EXIT
if [ -d campaigns/{CAMPAIGN_SLUG}/runtime ]; then
  find campaigns/{CAMPAIGN_SLUG}/runtime -type f >> "$TMP_LIST"
fi
if [ -d results/{CAMPAIGN_SLUG} ]; then
  find results/{CAMPAIGN_SLUG} -type f \\( -name run.log -o -name full_config.yaml \\) >> "$TMP_LIST"
fi
if [ -d tb_logs/{CAMPAIGN_SLUG} ]; then
  find tb_logs/{CAMPAIGN_SLUG} -type f -path "*/extracted/*" >> "$TMP_LIST"
fi
sort -u "$TMP_LIST" -o "$TMP_LIST"
if [ ! -s "$TMP_LIST" ]; then
  exit 0
fi
tar -czf - -T "$TMP_LIST"
"""

    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp_fh:
        archive_path = Path(tmp_fh.name)

    try:
        with archive_path.open("wb") as archive_fh:
            result = subprocess.run(
                [
                    "ssh",
                    "-p",
                    str(args.port),
                    args.host,
                    "bash",
                    "-lc",
                    remote_cmd,
                ],
                cwd=REPO_ROOT,
                stdout=archive_fh,
                stderr=subprocess.PIPE,
                check=False,
            )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.decode("utf-8", errors="replace"))
        if archive_path.stat().st_size == 0:
            print("No remote benchmark artifacts available yet.")
            return

        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(REPO_ROOT)
        print(f"Fetched compact artifacts into {REPO_ROOT}")
    finally:
        archive_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
