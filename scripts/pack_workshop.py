"""workshop-assets/의 실습 파일로 사이트 다운로드 파일을 다시 만듭니다.

사용: python scripts/pack_workshop.py
만드는 파일(public/downloads/):
  lora-workshop.zip      ZIP 안의 lora-workshop/ 폴더에 실습 파일 10개
  README.ko.md           사이트에서 바로 읽는 실습 안내
npm run build를 하면 public/downloads/가 dist/downloads/로 복사됩니다.
"""
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "workshop-assets"
DOWNLOADS = ROOT / "public" / "downloads"
FILES = [
    "lora_workshop.ipynb", "train.py", "lab_view.py", "guide.json", "requirements.txt",
    "train.jsonl", "validation.jsonl", "test.jsonl", "knowledge.json", "README.ko.md",
]

DOWNLOADS.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(DOWNLOADS / "lora-workshop.zip", "w", zipfile.ZIP_DEFLATED) as archive:
    for name in FILES:
        archive.write(ASSETS / name, f"lora-workshop/{name}")
shutil.copyfile(ASSETS / "README.ko.md", DOWNLOADS / "README.ko.md")
# 노트북은 웹 슬라이드가 코드 블록으로 보여 주므로 따로 내려받게 하지 않습니다(ZIP 안에만 포함).
print(f"{DOWNLOADS / 'lora-workshop.zip'}: {len(FILES)}개 파일")
