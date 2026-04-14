from datasets import load_dataset
from pathlib import Path
import re

#TARGET = "Tesla Model S Sedan 2012"
OUTDIR = Path("/data2/data/Tesla")
OUTDIR.mkdir(parents=True, exist_ok=True)

ds = load_dataset("Donghyun99/Stanford-Cars", cache_dir="/data/hf_cache")

def safe_name(s: str) -> str:
    # 파일/폴더명 안전하게 만들기
    s = s.strip()
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)

total = 0
for split in ds.keys():  # 'train', 'test'
    split_ds = ds[split]
    feat = split_ds.features.get("label", None)

    for i, ex in enumerate(split_ds):
        # label을 문자열로 변환
        val = ex["label"]
        if hasattr(feat, "int2str") and isinstance(val, int):
            lbl = feat.int2str(val)
        else:
            lbl = str(val)

        cls_dir = OUTDIR / split / safe_name(lbl)
        cls_dir.mkdir(parents=True, exist_ok=True)

        ex["image"].save(cls_dir / f"{i:06d}.jpg", quality=95)
        total += 1

print("saved:", total, "images to", OUTDIR)

