"""
GTSRB download, extract, convert to PNG, split train/test and output JSON
------------------------------------------------------------------------
Manual mode:
    1. Download Kaggle PNG version and extract to data/raw/archive/
       data/raw/archive/Train/0/*.png
    2. python src/prepare_data.py          # Only do conversion and splitting

Auto mode (download official PPM version):
    python src/prepare_data.py --auto
"""
import argparse, json, shutil, urllib.request
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split

ROOT     = Path(__file__).resolve().parents[1]
RAW_DIR  = ROOT / "data/raw"
IMG_DIR  = ROOT / "data/images"
ANN_DIR  = ROOT / "data/annotations"
URL      = "https://benchmark.ini.rub.de/Dataset/GTSRB_Final_Training_Images.zip"

def download_extract():
    """Download and extract official .ppm version to data/raw/"""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = RAW_DIR / "gtsrb.zip"
    if not zip_path.exists():
        print("[INFO] Downloading GTSRB...")
        urllib.request.urlretrieve(URL, zip_path)
    print("[INFO] Extracting...")
    shutil.unpack_archive(zip_path, RAW_DIR)

def convert_split(seed=42, test_ratio=0.2):
    """Convert Train data to data/images and split into train/test JSON"""
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    ANN_DIR.mkdir(parents=True, exist_ok=True)

    # Kaggle PNG version path
    train_dir = RAW_DIR / "archive" / "Train"
    # If not exists, check official PPM version extracted path
    if not train_dir.exists():
        train_dir = RAW_DIR / "GTSRB/Final_Training/Images"
    if not train_dir.exists():
        raise FileNotFoundError(
            "Cannot find Train folder, please confirm extraction to data/raw/ or use --auto")

    images, labels = [], []
    for class_dir in sorted(train_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        class_id = int(class_dir.name)
        for img_file in class_dir.glob("*.*"):          # Can grab .png or .ppm
            new_name = f"{class_id:02d}_{img_file.stem}.png"
            save_path = IMG_DIR / new_name
            if not save_path.exists():                 # Avoid duplicate conversion
                Image.open(img_file).convert("RGB").save(save_path)
            images.append(str(save_path.relative_to(ROOT)))
            labels.append(class_id)

    X_tr, X_te, y_tr, y_te = train_test_split(
        images, labels, test_size=test_ratio,
        stratify=labels, random_state=seed)

    for split, X, y in [("train", X_tr, y_tr), ("test", X_te, y_te)]:
        with open(ANN_DIR / f"{split}.json", "w") as f:
            json.dump([{"img": i, "label": l} for i, l in zip(X, y)], f, indent=2)

    print(f"Train: {len(X_tr)} images\nTest:  {len(X_te)} images")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--auto", action="store_true",
                        help="Automatically download official PPM version and extract")
    args = parser.parse_args()

    if args.auto:
        download_extract()
    convert_split()
