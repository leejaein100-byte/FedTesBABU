from pathlib import Path
import shutil

SRC_ROOT = Path("/data2/data/StanfordCars")
TRAIN_DIR = SRC_ROOT / "train"
TEST_DIR  = SRC_ROOT / "test"

# 통합 디렉터리(안전하게 새 폴더 추천)
OUT_ROOT = SRC_ROOT / "integrated"   # <- 원하시면 SRC_ROOT로 바꿔도 되지만 권장X
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# 어떤 확장자를 옮길지
EXTS = {".jpg", ".jpeg", ".png"}

def integrate_split(split_name: str, split_dir: Path, move: bool = False):
    """
    split_dir 아래의 <label> 디렉터리들을 OUT_ROOT/<label>/로 통합.
    move=False면 복사(copy2), True면 이동(move).
    """
    if not split_dir.exists():
        raise FileNotFoundError(f"Missing split dir: {split_dir}")

    n_files = 0
    n_labels = 0

    for label_dir in split_dir.iterdir():
        if not label_dir.is_dir():
            continue
        n_labels += 1

        out_label_dir = OUT_ROOT / label_dir.name
        out_label_dir.mkdir(parents=True, exist_ok=True)

        # label_dir 내부를 재귀적으로 탐색(혹시 하위 폴더가 있어도 처리)
        for src in label_dir.rglob("*"):
            if not src.is_file():
                continue
            if src.suffix.lower() not in EXTS:
                continue

            # 파일명 충돌 방지: train_ / test_ prefix
            dst = out_label_dir / f"{split_name}_{src.name}"

            # 만약 같은 이름이 이미 있으면 숫자 suffix 부여
            if dst.exists():
                stem = dst.stem
                suffix = dst.suffix
                k = 1
                while True:
                    cand = out_label_dir / f"{stem}__{k}{suffix}"
                    if not cand.exists():
                        dst = cand
                        break
                    k += 1

            if move:
                shutil.move(str(src), str(dst))
            else:
                shutil.copy2(str(src), str(dst))

            n_files += 1

    print(f"[{split_name}] labels={n_labels}, files={'moved' if move else 'copied'}={n_files}")

# 실행: 기본은 복사(원본 보존)
integrate_split("train", TRAIN_DIR, move=False)
integrate_split("test",  TEST_DIR,  move=False)

print("Integrated dataset saved to:", OUT_ROOT)
