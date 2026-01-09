import argparse
import tarfile
import urllib.request
from pathlib import Path


DEFAULT_URL = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar"


def parse_args():
    parser = argparse.ArgumentParser(description="Download MSD Task01_BrainTumour dataset")
    parser.add_argument("--url", default=DEFAULT_URL, help="Dataset tar URL")
    parser.add_argument(
        "--output-dir",
        default="data/raw/msd_task01",
        help="Directory to extract Task01_BrainTumour into",
    )
    parser.add_argument("--force", action="store_true", help="Re-download and extract even if present")
    return parser.parse_args()


def download_file(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    print(f"Downloading {url} to {dest}")
    urllib.request.urlretrieve(url, dest)


def extract_archive(archive_path: Path, output_dir: Path):
    print(f"Extracting {archive_path} to {output_dir}")
    with tarfile.open(archive_path, "r:*") as tar:
        tar.extractall(path=output_dir)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    archive_path = output_dir / "Task01_BrainTumour.tar"
    extracted_root = output_dir / "Task01_BrainTumour"

    if extracted_root.exists() and not args.force:
        print(f"Dataset already present at {extracted_root}")
        return

    download_file(args.url, archive_path)
    extract_archive(archive_path, output_dir)
    print(f"Ready: {extracted_root}")


if __name__ == "__main__":
    main()
