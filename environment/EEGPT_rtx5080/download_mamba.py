"""Download the official release in ranges and verify its published SHA256."""

import concurrent.futures
import hashlib
import json
import subprocess
from pathlib import Path

asset = json.loads(Path(__file__).with_name("mamba-asset.json").read_text())
cache = Path.home() / ".cache" / "eegpt-wheels"
parts = cache / ".mamba-parts"
parts.mkdir(parents=True, exist_ok=True)
chunk_size = 16 * 1024 * 1024


def download(start):
    end = min(start + chunk_size, asset["size"]) - 1
    destination = parts / str(start)
    expected_size = end - start + 1
    if not destination.exists() or destination.stat().st_size != expected_size:
        subprocess.run(
            [
                "curl", "-sSfL", "--retry", "5", "--retry-all-errors",
                "--retry-delay", "2", "--connect-timeout", "15",
                "--max-time", "360", "-r", f"{start}-{end}",
                "-o", str(destination),
                asset["browser_download_url"] + f"?eegpt_chunk={start}",
            ],
            check=True,
        )
    if destination.stat().st_size != expected_size:
        raise RuntimeError(f"Unexpected size for range {start}-{end}")
    return destination


starts = list(range(0, asset["size"], chunk_size))
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
    futures = [pool.submit(download, start) for start in starts]
    for count, future in enumerate(concurrent.futures.as_completed(futures), 1):
        future.result()
        print(f"Downloaded {count}/{len(starts)} ranges", flush=True)

digest = hashlib.sha256()
temporary = cache / (asset["name"] + ".verified-download")
with temporary.open("wb") as output:
    for start in starts:
        with (parts / str(start)).open("rb") as source:
            while block := source.read(1024 * 1024):
                digest.update(block)
                output.write(block)
expected_digest = asset["digest"].removeprefix("sha256:")
assert digest.hexdigest() == expected_digest, "Release SHA256 mismatch"
temporary.replace(cache / asset["name"])
print("Official Mamba release SHA256 verified", flush=True)
