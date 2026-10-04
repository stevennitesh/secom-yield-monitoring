"""Fetch the official UCI SECOM inputs with exact-file verification."""

from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile
import hashlib
import ssl
import certifi

from _script_path import ensure_src_on_path

ensure_src_on_path()
from secom.provenance import SECOM_FILE_SHA256, UCI_ARCHIVE_URL


def fetch_secom(output_dir: Path) -> None:
    """Reuse verified inputs, never overwrite a different local dataset."""
    missing = []
    for name, sha in SECOM_FILE_SHA256.items():
        target = output_dir / name
        if target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest() != sha:
                raise ValueError(f"Existing {target} differs from the reference; choose a different output directory")
        else:
            missing.append(name)
    if not missing:
        print("SECOM input hashes verified (existing files).")
        return
    with urlopen(UCI_ARCHIVE_URL, timeout=60, context=ssl.create_default_context(cafile=certifi.where())) as response:
        archive_bytes = response.read()
    # Read only the two exact members; do not extract arbitrary paths from the ZIP.
    with ZipFile(BytesIO(archive_bytes)) as archive:
        payloads = {name: archive.read(name) for name in missing}
    for name, payload in payloads.items():
        if hashlib.sha256(payload).hexdigest() != SECOM_FILE_SHA256[name]:
            raise ValueError(f"Downloaded {name} does not match the reference SHA-256")
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in payloads.items():
        with (output_dir / name).open("xb") as handle:
            handle.write(payload)
    print("SECOM input files downloaded and hashes verified.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    fetch_secom(args.output_dir)


if __name__ == "__main__":
    main()
