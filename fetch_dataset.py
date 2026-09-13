"""Download the original Cheng brain tumor dataset from figshare.

The Kaggle PNG mirror of this dataset drops the patient IDs and tumor-type
labels, which the patient-disjoint folds and the type-conditioned model both
need. These are the originals.

    https://doi.org/10.6084/m9.figshare.1512427.v5   (CC BY 4.0)

Four .zip parts, about 880 MB in total, plus cvind.mat (the authors' own 5-fold
cross-validation indices) and a README. prepare_dataset.py reads the .zip files
directly, so there is no need to unzip them.

    python fetch_dataset.py --out data/raw
    python fetch_dataset.py --out data/raw --skip-existing

Stdlib only. File ids are resolved from the figshare API at run time rather than
hard-coded, so a new dataset version does not silently download the old one.
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

ARTICLE_ID = 1512427
API_URL = f"https://api.figshare.com/v2/articles/{ARTICLE_ID}/files"

#: Used only if the API is unreachable. Verified against version 5.
FALLBACK_FILES = [
    {"name": "brainTumorDataPublic_1-766.zip", "id": 3381290, "size": 214401279},
    {"name": "brainTumorDataPublic_767-1532.zip", "id": 3381296, "size": 217848429},
    {"name": "brainTumorDataPublic_1533-2298.zip", "id": 3381293, "size": 215563856},
    {"name": "brainTumorDataPublic_2299-3064.zip", "id": 3381302, "size": 231679762},
    {"name": "cvind.mat", "id": 7005344, "size": 5736},
    {"name": "README 2024.txt", "id": 51340418, "size": 3303},
]

DOWNLOAD_URL = "https://ndownloader.figshare.com/files/{file_id}"


def human(size):
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024.0


def list_files(timeout=30):
    """Ask figshare what this article contains; fall back to verified ids."""
    try:
        with urllib.request.urlopen(API_URL, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        files = [{"name": f["name"], "id": f["id"], "size": f.get("size", 0)}
                 for f in payload]
        if files:
            return files, "figshare API"
    except (urllib.error.URLError, ValueError, KeyError, TimeoutError) as exc:
        print(f"Could not reach the figshare API ({exc}); using verified ids.")
    return FALLBACK_FILES, "built-in list"


def download(entry, out_dir, skip_existing=True, chunk=1 << 20):
    """Fetch one file, skipping it when a complete copy is already present."""
    target = os.path.join(out_dir, entry["name"])
    expected = entry.get("size") or 0

    if os.path.exists(target):
        actual = os.path.getsize(target)
        if skip_existing and (not expected or actual == expected):
            print(f"  have  {entry['name']}  ({human(actual)})")
            return target
        print(f"  redo  {entry['name']}  (have {human(actual)}, "
              f"expected {human(expected)})")

    url = DOWNLOAD_URL.format(file_id=entry["id"])
    partial = target + ".part"
    started = time.time()
    got = 0

    with urllib.request.urlopen(url, timeout=60) as response, \
            open(partial, "wb") as fh:
        while True:
            block = response.read(chunk)
            if not block:
                break
            fh.write(block)
            got += len(block)
            if expected:
                pct = 100.0 * got / expected
                sys.stdout.write(f"\r  get   {entry['name']}  {pct:5.1f}%  "
                                 f"{human(got)}/{human(expected)}")
                sys.stdout.flush()

    if expected and got != expected:
        os.remove(partial)
        raise RuntimeError(
            f"{entry['name']} came down truncated: {got} of {expected} bytes. "
            f"Re-run to retry."
        )

    os.replace(partial, target)          # only a complete file gets the real name
    elapsed = time.time() - started
    sys.stdout.write(f"\r  done  {entry['name']}  {human(got)} in {elapsed:.0f}s\n")
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="data/raw",
                        help="where to put the archives (default: data/raw)")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--force", action="store_true",
                        help="re-download even if a complete copy exists")
    parser.add_argument("--zips-only", action="store_true",
                        help="skip cvind.mat and the README")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    files, source = list_files()

    if args.zips_only:
        files = [f for f in files if f["name"].lower().endswith(".zip")]

    total = sum(f.get("size", 0) for f in files)
    print(f"{len(files)} file(s), {human(total)} total, listed via {source}")
    print(f"Destination: {os.path.abspath(args.out)}\n")

    for entry in files:
        download(entry, args.out, skip_existing=not args.force)

    print(f"\nNext:\n"
          f"    python prepare_dataset.py --src {args.out} --out data/ --limit 20\n"
          f"    python prepare_dataset.py --src {args.out} --out data/\n"
          f"    python folds.py --manifest data/manifest.csv --out data/folds.csv")


if __name__ == "__main__":
    main()
