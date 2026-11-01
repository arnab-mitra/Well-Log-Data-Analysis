"""Download the published FORCE 2020 LAS archive with integrity checks."""

from __future__ import annotations

import hashlib
import shutil
import tempfile
import urllib.request
from collections.abc import Callable
from pathlib import Path

FORCE_ZENODO_RECORD = "4351156"
FORCE_ARCHIVE_NAME = "LAS_files_Force_2020_all_wells_train_test_blind_hidden_final.zip"
FORCE_ARCHIVE_URL = (
    f"https://zenodo.org/api/records/{FORCE_ZENODO_RECORD}/files/"
    f"{FORCE_ARCHIVE_NAME}/content"
)
FORCE_ARCHIVE_MD5 = "da0d72d5783a780ac71779453d9fbfbf"


def md5sum(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the lowercase MD5 digest of a file."""
    digest = hashlib.md5(usedforsecurity=False)
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_force_archive(
    destination: str | Path,
    *,
    url: str = FORCE_ARCHIVE_URL,
    expected_md5: str = FORCE_ARCHIVE_MD5,
    overwrite: bool = False,
    timeout: float = 120.0,
    opener: Callable[..., object] = urllib.request.urlopen,
) -> Path:
    """Download the FORCE archive atomically and verify its published MD5.

    An existing valid file is reused. Invalid existing files are rejected unless
    ``overwrite`` is true. A failed transfer never leaves a partial destination.
    """
    destination = Path(destination).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected = expected_md5.lower().removeprefix("md5:")

    if destination.exists() and not overwrite:
        actual = md5sum(destination)
        if actual == expected:
            return destination
        raise ValueError(
            f"Existing file has MD5 {actual}, expected {expected}: {destination}"
        )

    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, prefix=f".{destination.name}.", suffix=".part", delete=False
        ) as output:
            temporary = Path(output.name)
            with opener(url, timeout=timeout) as response:
                shutil.copyfileobj(response, output, length=1024 * 1024)
        actual = md5sum(temporary)
        if actual != expected:
            raise ValueError(f"Downloaded file has MD5 {actual}, expected {expected}")
        temporary.replace(destination)
        return destination
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
