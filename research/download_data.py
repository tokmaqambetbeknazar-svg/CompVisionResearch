"""Download pinned public pose tables; check every file's SHA-256."""
import hashlib
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
COMMIT = "71b31a8853f9d9b5568d88484b4e3263ddc5da03"
BASE = f"https://raw.githubusercontent.com/YattaHab/REHAB24-6-Standardized-Training-Data/{COMMIT}"
FILES = {
    "rehab246_pushup_positions_standardized.csv": (
        BASE + "/standardized_data/rehab246_pushup_positions_standardized.csv",
        "7f688b6debf297e7305074ce2010fb4506196a28a874243fd3c0b5487101f512"),
    "rehab246_squat_positions_standardized.csv": (
        BASE + "/standardized_data/rehab246_squat_positions_standardized.csv",
        "83be97dbc1dd6abc96b9a4c03d173917aee11c906a1fc918420c226bff114106"),
    "original_segmentation.csv": (
        "https://zenodo.org/records/13305826/files/Segmentation.csv?download=1",
        "72c70bc7e56eafb132452a850f7a56d21c3cb4a602bab1e98370f9b606567100"),
}


def download():
    folder = ROOT / "data"
    folder.mkdir(exist_ok=True)
    for name, (url, expected) in FILES.items():
        path = folder / name
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == expected:
            print("Verified:", name)
            continue
        print("Downloading:", name)
        request = Request(url, headers={"User-Agent": "FitnessCourseResearch/1.0"})
        with urlopen(request, timeout=120) as response:
            payload = response.read()
        if hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError(f"Unexpected file checksum: {name}; source may have changed")
        path.write_bytes(payload)
    return folder


if __name__ == "__main__":
    download()
