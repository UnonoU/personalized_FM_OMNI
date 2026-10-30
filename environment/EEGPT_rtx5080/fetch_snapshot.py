"""Cache exact PyPI artifacts for CPython 3.10, verifying published hashes."""

import concurrent.futures
import hashlib
import json
import urllib.request
import urllib.parse
from pathlib import Path

from packaging.tags import compatible_tags, cpython_tags, platform_tags
from packaging.utils import parse_wheel_filename

HERE = Path(__file__).resolve().parent
CACHE = Path.home() / ".cache" / "eegpt-wheels"
CACHE.mkdir(parents=True, exist_ok=True)
platforms = list(platform_tags())
tags = list(cpython_tags((3, 10), abis=["cp310"], platforms=platforms))
tags += list(compatible_tags((3, 10), interpreter="cp310", platforms=platforms))
priorities = {tag: index for index, tag in reversed(list(enumerate(tags)))}


def file_sha256(path):
    result = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def choose_artifact(artifacts):
    candidates = []
    for artifact in artifacts:
        if not artifact["filename"].endswith(".whl"):
            continue
        wheel_tags = parse_wheel_filename(artifact["filename"])[3]
        scores = [priorities[tag] for tag in wheel_tags if tag in priorities]
        if scores:
            candidates.append((min(scores), artifact))
    if candidates:
        return min(candidates, key=lambda item: item[0])[1]
    return next(item for item in artifacts if item["packagetype"] == "sdist")


def fetch(pin):
    name, version = pin.split("==")
    for attempt in range(4):
        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({})
            if attempt % 2 else urllib.request.ProxyHandler()
        )
        try:
            with opener.open(
                f"https://pypi.org/pypi/{name}/{version}/json", timeout=15
            ) as response:
                artifact = choose_artifact(json.load(response)["urls"])
            break
        except Exception:
            if attempt == 3:
                raise
    destination = CACHE / artifact["filename"]
    digest = artifact["digests"]["sha256"]
    if destination.exists() and file_sha256(destination) == digest:
        return artifact
    temporary = destination.with_name(destination.name + ".download")
    last_error = None
    for attempt in range(4):
        # The mirror serves the same artifacts; always verify PyPI's hash.
        mirror = attempt in (0, 3)
        url = (
            "https://pypi.tuna.tsinghua.edu.cn"
            + urllib.parse.urlsplit(artifact["url"]).path
            if mirror else artifact["url"]
        )
        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({})
            if attempt != 2 else urllib.request.ProxyHandler()
        )
        try:
            actual_digest = hashlib.sha256()
            with opener.open(url, timeout=20) as source:
                with temporary.open("wb") as output:
                    while chunk := source.read(1024 * 1024):
                        actual_digest.update(chunk)
                        output.write(chunk)
            if actual_digest.hexdigest() != digest:
                raise RuntimeError("PyPI artifact SHA256 mismatch")
            temporary.replace(destination)
            artifact["downloaded_from"] = url
            return artifact
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"{pin}: {last_error}")


pins = [
    line for line in (HERE / "requirements.txt").read_text().splitlines()
    if line and not line.startswith("#")
]
artifacts = []
errors = []
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
    futures = {pool.submit(fetch, pin): pin for pin in pins}
    for count, future in enumerate(concurrent.futures.as_completed(futures), 1):
        try:
            artifact = future.result()
        except Exception as exc:
            errors.append(f"{futures[future]}: {exc}")
            print(f"FAILED {errors[-1]}", flush=True)
            continue
        artifacts.append(artifact)
        print(f"Verified {count}/{len(pins)}: {artifact['filename']}", flush=True)
(HERE / "pypi-artifacts.json").write_text(
    json.dumps(artifacts, indent=2, ensure_ascii=False) + "\n"
)
if errors:
    raise RuntimeError("\n".join(errors))
