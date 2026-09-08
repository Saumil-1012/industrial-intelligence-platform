"""
Download M5 Forecasting dataset from Kaggle.
Requires: KAGGLE_USERNAME and KAGGLE_KEY set in .env or environment.

Usage:
    python data/supply_chain/download.py
"""

import os
import subprocess
import zipfile
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(__file__).parent


def download_m5():
    """Download and extract M5 dataset via Kaggle CLI."""
    kaggle_user = os.getenv("KAGGLE_USERNAME")
    kaggle_key = os.getenv("KAGGLE_KEY")

    if not kaggle_user or not kaggle_key:
        raise EnvironmentError(
            "Set KAGGLE_USERNAME and KAGGLE_KEY in your .env file.\n"
            "Get your API token at: https://www.kaggle.com/settings"
        )

    os.environ["KAGGLE_USERNAME"] = kaggle_user
    os.environ["KAGGLE_KEY"] = kaggle_key

    print("Downloading M5 Forecasting dataset...")
    subprocess.run(
        [
            "kaggle", "competitions", "download",
            "-c", "m5-forecasting-accuracy",
            "-p", str(DATA_DIR),
        ],
        check=True,
    )

    zip_path = DATA_DIR / "m5-forecasting-accuracy.zip"
    if zip_path.exists():
        print("Extracting...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(DATA_DIR)
        zip_path.unlink()
        print(f"Done. Files in {DATA_DIR}:")
        for f in DATA_DIR.iterdir():
            print(f"  {f.name}")
    else:
        print("Download may have failed — check Kaggle credentials.")


if __name__ == "__main__":
    download_m5()
