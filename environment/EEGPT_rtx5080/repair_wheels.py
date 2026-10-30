"""Repair known upstream WHEEL metadata defects without changing payloads.

Decord: https://github.com/dmlc/decord/issues/356
CMake 3.26.1: a blank line separates Tag headers from the other headers.
Original PyPI wheels and SHA256 values remain intact in the download cache.
"""

import hashlib
import json
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
CACHE = Path.home() / ".cache" / "eegpt-wheels"
REPAIRED = CACHE / "repaired"
REPAIRED.mkdir(parents=True, exist_ok=True)
artifacts = json.loads((HERE / "pypi-artifacts.json").read_text())
records = []


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


for name in ("cmake", "decord"):
    artifact = next(a for a in artifacts if a["filename"].startswith(name + "-"))
    original = CACHE / artifact["filename"]
    if not original.exists():
        url = "https://pypi.tuna.tsinghua.edu.cn" + urllib.parse.urlsplit(
            artifact["url"]
        ).path
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(url, timeout=60) as source:
            original.write_bytes(source.read())
    assert digest(original) == artifact["digests"]["sha256"]
    with tempfile.TemporaryDirectory(prefix="eegpt-wheel-repair-") as temporary:
        unpacked = Path(temporary) / "unpacked"
        if name == "decord":
            # The upstream RECORD also mis-hashes top_level.txt. The complete
            # archive was verified against PyPI above; wheel pack regenerates
            # RECORD, and all other files are byte-compared below.
            source = unpacked / "decord-0.6.0"
            with zipfile.ZipFile(original) as archive:
                archive.extractall(source)
        else:
            subprocess.run(
                [sys.executable, "-m", "wheel", "unpack", str(original),
                 "-d", str(unpacked)], check=True,
            )
        source = next(unpacked.iterdir())
        metadata = next(source.glob("*.dist-info/WHEEL"))
        before = metadata.read_text()
        if name == "cmake":
            assert "\n\nTag:" in before
            after = "\n".join(line for line in before.splitlines() if line) + "\n"
        else:
            assert "Tag: cp36-cp36m-manylinux2010_x86_64" in before
            after = before.replace(
                "Tag: cp36-cp36m-manylinux2010_x86_64",
                "Tag: py3-none-manylinux2010_x86_64",
            )
        metadata.write_text(after)
        subprocess.run(
            [sys.executable, "-m", "wheel", "pack", str(source),
             "--build-number", "1eegpt", "-d", str(REPAIRED)], check=True,
        )
    repaired = next(REPAIRED.glob(name + "-*-1eegpt-*.whl"))
    with zipfile.ZipFile(original) as old, zipfile.ZipFile(repaired) as new:
        original_files = {info.filename for info in old.infolist() if not info.is_dir()}
        repaired_files = {info.filename for info in new.infolist() if not info.is_dir()}
        assert original_files == repaired_files
        for filename in original_files:
            if filename.endswith((".dist-info/WHEEL", ".dist-info/RECORD")):
                continue
            assert old.read(filename) == new.read(filename), filename
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-index", "--no-deps",
         "--force-reinstall", "--disable-pip-version-check", str(repaired)],
        check=True,
    )
    records.append({
        "package": name, "source_url": artifact["url"],
        "source_sha256": digest(original), "repaired_wheel": repaired.name,
        "repaired_sha256": digest(repaired),
        "payload_identical": True, "original_WHEEL": before,
        "corrected_WHEEL_before_repack": after,
    })
(HERE / "wheel-repairs.json").write_text(json.dumps(records, indent=2) + "\n")
