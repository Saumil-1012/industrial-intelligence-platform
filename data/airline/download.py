"""
Download DOT Airline On-Time Performance dataset from Kaggle.
Requires: KAGGLE_USERNAME and KAGGLE_KEY set in .env

Usage:
    python data/airline/download.py
"""

import os
import subprocess
import zipfile
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(__file__).parent


def download_dot_airline():
    kaggle_user = os.getenv("KAGGLE_USERNAME")
    kaggle_key  = os.getenv("KAGGLE_KEY")

    if not kaggle_user or not kaggle_key:
        raise EnvironmentError(
            "Set KAGGLE_USERNAME and KAGGLE_KEY in your .env file.\n"
            "Get your API token at: https://www.kaggle.com/settings"
        )

    os.environ["KAGGLE_USERNAME"] = kaggle_user
    os.environ["KAGGLE_KEY"]      = kaggle_key

    print("Downloading DOT Airline On-Time Performance dataset...")
    subprocess.run([
        "kaggle", "datasets", "download",
        "-d", "yuanyuwendymu/airline-delay-and-cancellation-data-2009-2018",
        "-p", str(DATA_DIR),
    ], check=True)

    zip_path = DATA_DIR / "airline-delay-and-cancellation-data-2009-2018.zip"
    if zip_path.exists():
        print("Extracting...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(DATA_DIR)
        zip_path.unlink()
        print(f"Done. Files in {DATA_DIR}:")
        for f in sorted(DATA_DIR.iterdir()):
            print(f"  {f.name} ({f.stat().st_size / 1e6:.1f} MB)")
    else:
        print("Download may have failed — check Kaggle credentials.")


if __name__ == "__main__":
    download_dot_airline()
