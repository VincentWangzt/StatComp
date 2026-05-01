from __future__ import annotations

import argparse
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CAMPAIGN_SLUG = "default_config_grid"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch compact runtime metadata from the remote server for inspection. "
            "Final figures and tables should be generated, committed, pushed, and pulled through git."
        )
    )
    parser.add_argument("--host", default="root@connect.nmb1.seetacloud.com")
    parser.add_argument("--port", type=int, default=48236)
    parser.add_argument("--remote-repo", default="~/ruivi")
    parser.add_argument("--campaign-slug", default=DEFAULT_CAMPAIGN_SLUG)
    parser.add_argument(
        "--remote-artifact-root",
        default=None,
        help="Optional remote root containing results/ and tb_logs/. Use for campaigns stored outside the repo.",
    )
    args = parser.parse_args()

    if shutil.which("ssh") is None:
        raise RuntimeError("ssh command not found on PATH")

    artifact_root = args.remote_artifact_root or args.remote_repo
    remote_cmd = f"""
set -e
REMOTE_REPO={args.remote_repo}
ARTIFACT_ROOT={artifact_root}
STAGE="$(mktemp -d)"
cleanup() {{
  rm -rf "$STAGE"
}}
trap cleanup EXIT
cd "$REMOTE_REPO"
if [ -d campaigns/{args.campaign_slug}/runtime ]; then
  mkdir -p "$STAGE"
  tar -cf - campaigns/{args.campaign_slug}/runtime | tar -xf - -C "$STAGE"
fi

TMP_LIST="$(mktemp)"
cd "$ARTIFACT_ROOT"
if [ -d results/{args.campaign_slug} ]; then
  find results/{args.campaign_slug} -type f \\( -name run.log -o -name full_config.yaml \\) >> "$TMP_LIST"
fi
if [ -d tb_logs/{args.campaign_slug} ]; then
  find tb_logs/{args.campaign_slug} -type f -path "*/extracted/*" >> "$TMP_LIST"
fi
sort -u "$TMP_LIST" -o "$TMP_LIST"
if [ -s "$TMP_LIST" ]; then
  tar -cf - -T "$TMP_LIST" | tar -xf - -C "$STAGE"
fi
rm -f "$TMP_LIST"
if [ -z "$(find "$STAGE" -type f -print -quit)" ]; then
  exit 0
fi
cd "$STAGE"
tar -czf - .
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
