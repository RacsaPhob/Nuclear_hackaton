"""Download the official DIV2K HR archives with resumable transfers."""

import argparse
from pathlib import Path
import time
import urllib.error
import urllib.request
import zipfile


BASE_URL = "https://data.vision.ee.ethz.ch/cvl/DIV2K/"
ARCHIVES = ("DIV2K_train_HR.zip", "DIV2K_valid_HR.zip")


def download_archive(name: str, destination: Path) -> None:
    url = BASE_URL + name
    with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as response:
        expected_size = int(response.headers["Content-Length"])
    if destination.exists():
        if destination.stat().st_size != expected_size or not zipfile.is_zipfile(destination):
            raise ValueError(f"Existing archive is incomplete or invalid: {destination}")
        print(f"Already downloaded: {name}", flush=True)
        return

    partial = destination.with_suffix(".zip.part")
    for attempt in range(3):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset == expected_size and zipfile.is_zipfile(partial):
            partial.rename(destination)
            return
        if offset > expected_size:
            raise ValueError(f"Partial download exceeds the archive size: {partial}")
        request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                if response.status == 206:
                    content_range = response.headers.get("Content-Range", "")
                    if not content_range.startswith(f"bytes {offset}-"):
                        raise ValueError(f"Unexpected Content-Range: {content_range}")
                    mode = "ab"
                else:
                    offset = 0
                    mode = "wb"
                last_report = 0.0
                with partial.open(mode) as output:
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
                        offset += len(chunk)
                        now = time.monotonic()
                        if now - last_report > 15:
                            print(f"{name}: {offset / expected_size:.1%}", flush=True)
                            last_report = now
            if partial.stat().st_size != expected_size:
                raise OSError("Transfer ended before the complete archive arrived.")
            if not zipfile.is_zipfile(partial):
                raise ValueError(f"Downloaded file is not a ZIP archive: {partial}")
            partial.rename(destination)
            print(f"Downloaded: {name}", flush=True)
            return
        except (OSError, urllib.error.URLError) as error:
            if attempt == 2:
                raise
            print(f"Retrying {name} after {error}", flush=True)
            time.sleep(2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for name in ARCHIVES:
        download_archive(name, args.output / name)


if __name__ == "__main__":
    main()
