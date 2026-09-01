"""Download the MobileNet SSD files required by the application."""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve


MODEL_DIR = Path(__file__).parent / "models"
MODEL_FILES = {
    "deploy.prototxt": (
        "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt"
    ),
    "mobilenet_iter_73000.caffemodel": (
        "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/"
        "mobilenet_iter_73000.caffemodel"
    ),
}


def main() -> None:
    """Download missing model files into the ignored models directory."""
    MODEL_DIR.mkdir(exist_ok=True)

    for filename, url in MODEL_FILES.items():
        destination = MODEL_DIR / filename
        if destination.exists():
            print(f"Already present: {destination}")
            continue

        print(f"Downloading {filename}...")
        urlretrieve(url, destination)
        print(f"Saved: {destination}")

    print("MobileNet SSD files are ready.")


if __name__ == "__main__":
    main()
