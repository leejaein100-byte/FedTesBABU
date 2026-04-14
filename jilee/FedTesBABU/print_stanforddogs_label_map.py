import argparse
import os
from types import SimpleNamespace

from utils.Stanford_Dog_args_iid_non_iid_non_overlapping import StanfordDogsDataset


def main():
    parser = argparse.ArgumentParser(description="Print StanfordDogsDataset label index-to-name mapping")
    parser.add_argument("--dataset", type=str, default="Stanford_dog")
    parser.add_argument("--use_bbox", action="store_true")
    parser.add_argument("--output_path", type=str, default=None)
    args = parser.parse_args()

    dataset_args = SimpleNamespace(dataset=args.dataset, use_bbox=args.use_bbox)
    dataset = StanfordDogsDataset(dataset_args, transform=None)

    output_path = args.output_path
    if output_path is None:
        suffix = "bbox" if args.use_bbox else "original"
        output_path = os.path.join(os.getcwd(), f"{args.dataset}_{suffix}_label_map.log")

    lines = []
    lines.append(f"dataset={args.dataset}")
    lines.append(f"use_bbox={args.use_bbox}")
    lines.append(f"images_dir={dataset.images_dir}")
    lines.append(f"num_classes={len(dataset.class_names)}")
    lines.append("")
    lines.append("Label index to class name mapping (0-based):")

    for idx, class_name in enumerate(dataset.class_names):
        lines.append(f"{idx}	{class_name}")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Saved label map to: {output_path}")


if __name__ == "__main__":
    main()
