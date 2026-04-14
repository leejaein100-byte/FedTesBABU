import argparse
import copy
import json
import os
import re
from types import SimpleNamespace

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

from Gr_model_with_cluster_cost import construct_TesNet
from util.helpers import makedir
from util.log import create_logger
from util.preprocess import mean, std, undo_preprocess_input_function
from utils.Stanford_Dog_args_iid_non_iid_non_overlapping import (
    load_Stan_data,
    setup_datasets,
)


def _get_nested(cfg, keys, default=None):
    for k in keys:
        if k in cfg:
            return cfg[k]
    return default


def _read_settings(model_dir):
    settings_path = os.path.join(model_dir, "settings.json")
    if os.path.isfile(settings_path):
        with open(settings_path, "r") as f:
            return json.load(f)
    return {}


def _read_seed_from_log(model_dir):
    train_log = os.path.join(model_dir, "train.log")
    if not os.path.isfile(train_log):
        return None

    seed_patterns = [
        re.compile(r"\bseed\b\s*[:=]\s*(\d+)", re.IGNORECASE),
        re.compile(r"\bSEED\b\s*[:=]\s*(\d+)", re.IGNORECASE),
    ]

    with open(train_log, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            for pattern in seed_patterns:
                m = pattern.search(line)
                if m:
                    return int(m.group(1))
    return None


def _build_data_args(cli_args, saved_cfg):
    seed_from_cfg = _get_nested(saved_cfg, ["seed"], None)
    seed_from_log = _read_seed_from_log(cli_args.model_dir)

    seed = cli_args.seed if cli_args.seed is not None else seed_from_cfg
    if seed is None:
        seed = seed_from_log if seed_from_log is not None else 42

    return SimpleNamespace(
        seed=seed,
        iid=bool(_get_nested(saved_cfg, ["iid"], cli_args.iid)),
        num_channels=int(_get_nested(saved_cfg, ["num_channels"], 64)),
        server_id_size=int(_get_nested(saved_cfg, ["server_id_size"], 0)),
        local_bs=int(_get_nested(saved_cfg, ["local_bs"], 128)),
        num_users=int(_get_nested(saved_cfg, ["num_users"], cli_args.num_users)),
        arch=str(_get_nested(saved_cfg, ["arch"], cli_args.arch)),
        dataset=str(_get_nested(saved_cfg, ["dataset"], cli_args.dataset)),
        SL_epochs=int(_get_nested(saved_cfg, ["SL_epochs"], 1)),
        fine_tune_epochs=int(_get_nested(saved_cfg, ["fine_tune_epochs"], 1)),
        alpha=float(_get_nested(saved_cfg, ["alpha"], cli_args.alpha)),
        use_bbox=bool(_get_nested(saved_cfg, ["use_bbox"], cli_args.use_bbox)),
        temp=float(_get_nested(saved_cfg, ["temp"], 1.0)),
        warmup_ep=int(_get_nested(saved_cfg, ["warmup_ep"], 0)),
        hyperparam=float(_get_nested(saved_cfg, ["hyperparam"], -1e-6)),
        num_teachers=int(_get_nested(saved_cfg, ["num_teachers"], 1)),
        patch_num=int(_get_nested(saved_cfg, ["patch_num"], cli_args.patch_num)),
        score_logit=bool(_get_nested(saved_cfg, ["score logits", "score_logit"], False)),
        patch_div_loss=bool(_get_nested(saved_cfg, ["patch_div_loss"], False)),
        last_layer=bool(_get_nested(saved_cfg, ["last layer", "last_layer"], False)),
        cons_mode=str(_get_nested(saved_cfg, ["cons_mode"], "polar")),
        Tscale=bool(_get_nested(saved_cfg, ["Tscale"], False)),
        img_size=int(_get_nested(saved_cfg, ["img_size"], cli_args.img_size)),
        tr_frac=float(_get_nested(saved_cfg, ["tr_frac"], cli_args.tr_frac)),
        workers=int(_get_nested(saved_cfg, ["workers"], 4)),
    )


def _resolve_model_path(cli_args):
    if cli_args.model_path:
        return cli_args.model_path

    name = "client_model.pth" if cli_args.use_client_model else "final_model.pth"
    return os.path.join(cli_args.model_dir, name)


def _build_model(data_args, model_path, device):
    state_dict = torch.load(model_path, map_location=device)

    if "last_layer.weight" not in state_dict or "prototype_vectors" not in state_dict:
        raise RuntimeError("State dict is missing required keys: last_layer.weight / prototype_vectors")

    num_classes = int(state_dict["last_layer.weight"].shape[0])
    num_channels = int(state_dict["prototype_vectors"].shape[1])

    model = construct_TesNet(
        args=data_args,
        base_architecture=data_args.arch,
        prototype_per_class=3,
        dataset=data_args.dataset,
        pretrained=False,
        img_size=data_args.img_size,
        prototype_shape=(num_classes, num_channels, 1, 1),
        num_classes=num_classes,
        prototype_activation_function="log",
        add_on_layers_type="regular",
    )

    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()
    return model


def _save_preprocessed_img(path, preprocessed_img_batch, index=0):
    img_copy = copy.deepcopy(preprocessed_img_batch[index : index + 1])
    undo_preprocessed_img = undo_preprocess_input_function(img_copy)[0]
    undo_preprocessed_img = undo_preprocessed_img.detach().cpu().numpy()
    undo_preprocessed_img = np.transpose(undo_preprocessed_img, [1, 2, 0])
    plt.imsave(path, undo_preprocessed_img)
    return undo_preprocessed_img


def _imsave_with_bbox(fname, img_rgb, h0, h1, w0, w1, color=(0, 255, 255)):
    img_bgr_uint8 = cv2.cvtColor(np.uint8(255 * img_rgb), cv2.COLOR_RGB2BGR)
    cv2.rectangle(img_bgr_uint8, (w0, h0), (w1 - 1, h1 - 1), color, thickness=2)
    img_rgb_uint8 = img_bgr_uint8[..., ::-1]
    img_rgb_float = np.float32(img_rgb_uint8) / 255
    plt.imsave(fname, img_rgb_float)


def _save_activation_overlay(prefix_path, activation_2d, original_img, out_h, out_w):
    upsampled = cv2.resize(activation_2d, dsize=(out_w, out_h), interpolation=cv2.INTER_CUBIC)

    rescaled = upsampled - np.amin(upsampled)
    denom = np.amax(rescaled)
    if denom > 1e-12:
        rescaled = rescaled / denom
    else:
        rescaled = np.zeros_like(rescaled)

    heatmap = cv2.applyColorMap(np.uint8(255 * rescaled), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255.0
    heatmap = heatmap[..., ::-1]
    overlayed = 0.5 * original_img + 0.3 * heatmap

    np.save(f"{prefix_path}_upsampled.npy", upsampled)
    plt.imsave(f"{prefix_path}_heatmap.png", heatmap)
    plt.imsave(f"{prefix_path}_overlay.png", overlayed)


def _save_top_patch_artifacts(
    class_dir,
    class_rank,
    class_idx,
    patch_scores,
    feat_h,
    feat_w,
    img_h,
    img_w,
    original_img,
    topk_patches,
    log,
):
    k_patch = min(topk_patches, int(patch_scores.numel()))
    top_patch_scores, top_patch_indices = torch.topk(patch_scores, k=k_patch)

    patch_summary = []
    for patch_rank in range(k_patch):
        patch_idx = int(top_patch_indices[patch_rank].item())
        patch_score = float(top_patch_scores[patch_rank].item())

        h0, h1, w0, w1 = _patch_bbox(patch_idx, feat_h, feat_w, img_h, img_w)
        patch = original_img[h0:h1, w0:w1, :]

        patch_prefix = os.path.join(class_dir, f"top{patch_rank+1}_patch")
        plt.imsave(f"{patch_prefix}_crop.png", patch)
        _imsave_with_bbox(
            fname=f"{patch_prefix}_on_original.png",
            img_rgb=original_img,
            h0=h0,
            h1=h1,
            w0=w0,
            w1=w1,
            color=(0, 255, 255),
        )

        patch_mask = np.zeros((feat_h, feat_w), dtype=np.float32)
        row = int(patch_idx // feat_w)
        col = int(patch_idx % feat_w)
        patch_mask[row, col] = patch_score
        _save_activation_overlay(
            os.path.join(class_dir, f"top{patch_rank+1}_patch"),
            patch_mask,
            original_img,
            img_h,
            img_w,
        )

        log(
            f"Top-{class_rank} class {class_idx}, patch {patch_rank+1}: "
            f"flat_idx={patch_idx}, score={patch_score:.6f}, bbox=({h0}:{h1}, {w0}:{w1})"
        )
        patch_summary.append(
            {
                "rank": patch_rank + 1,
                "flat_idx": patch_idx,
                "score": patch_score,
            }
        )

    return patch_summary


def _load_from_fl_split(data_args, client_idx, sample_idx):
    dict_users, server_idx, dataset = setup_datasets(data_args)
    X_test, y_test = load_Stan_data(
        data_args,
        dataset,
        server_idx,
        client_idx,
        dict_users,
        train=False,
        private=True,
    )

    if len(X_test) == 0:
        raise RuntimeError(f"Client {client_idx} has empty test split.")

    if sample_idx < 0 or sample_idx >= len(X_test):
        raise IndexError(f"sample_idx={sample_idx} out of range (0..{len(X_test)-1})")

    image = X_test[sample_idx].unsqueeze(0)
    label = int(y_test[sample_idx].item())
    return image, label


def _load_from_global_test_split(data_args, sample_idx):
    dict_users, server_idx, dataset = setup_datasets(data_args)

    X_test_all = []
    y_test_all = []
    for client_idx in range(data_args.num_users):
        X_test, y_test = load_Stan_data(
            data_args,
            dataset,
            server_idx,
            client_idx,
            dict_users,
            train=False,
            private=True,
        )
        if len(X_test) == 0:
            continue
        X_test_all.append(X_test)
        y_test_all.append(y_test)

    if not X_test_all:
        raise RuntimeError("Global test split is empty.")

    X_test_global = torch.cat(X_test_all, dim=0)
    y_test_global = torch.cat(y_test_all, dim=0)

    if sample_idx < 0 or sample_idx >= len(X_test_global):
        raise IndexError(f"sample_idx={sample_idx} out of range (0..{len(X_test_global)-1})")

    image = X_test_global[sample_idx].unsqueeze(0)
    label = int(y_test_global[sample_idx].item())
    return image, label


def _load_from_image_path(data_args, image_path, image_label):
    normalize = transforms.Normalize(mean=mean, std=std)
    preprocess = transforms.Compose(
        [
            transforms.Resize((data_args.img_size, data_args.img_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )

    img_pil = Image.open(image_path).convert("RGB")
    img_tensor = preprocess(img_pil)
    image = img_tensor.unsqueeze(0)
    return image, image_label


def _get_class_names(data_args):
    if data_args.dataset == "Stanford_dog":
        images_dir = "/home/jilee/jilee/data/Cropped_Images" if data_args.use_bbox else "/home/jilee/jilee/data/Images"
    elif data_args.dataset == "Stanford_cars":
        images_dir = "/root/stanford_cars_cropped/integrated" if data_args.use_bbox else "/data2/data/StanfordCars/integrated"
    else:
        return None

    if not os.path.isdir(images_dir):
        return None

    return sorted(
        entry
        for entry in os.listdir(images_dir)
        if os.path.isdir(os.path.join(images_dir, entry))
    )


def _label_to_class_name(data_args, label):
    class_names = _get_class_names(data_args)
    if class_names is None or label < 0 or label >= len(class_names):
        return None
    return class_names[label]


def _patch_bbox(flat_idx, feat_h, feat_w, img_h, img_w):
    row = int(flat_idx // feat_w)
    col = int(flat_idx % feat_w)

    h0 = int(np.floor(row * img_h / feat_h))
    h1 = int(np.ceil((row + 1) * img_h / feat_h))
    w0 = int(np.floor(col * img_w / feat_w))
    w1 = int(np.ceil((col + 1) * img_w / feat_w))

    h0 = max(0, min(h0, img_h - 1))
    h1 = max(h0 + 1, min(h1, img_h))
    w0 = max(0, min(w0, img_w - 1))
    w1 = max(w0 + 1, min(w1, img_w))
    return h0, h1, w0, w1


def main():
    parser = argparse.ArgumentParser(description="FL-aware local analysis for final logits and top-3 patches")

    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--use_client_model", action="store_true")

    parser.add_argument("--client_idx", type=int, default=0)
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--use_global_dataset", action="store_true")

    parser.add_argument("--image_path", type=str, default=None)
    parser.add_argument("--image_label", type=int, default=-1)

    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--topk_classes", type=int, default=3)
    parser.add_argument("--topk_patches", type=int, default=3)

    parser.add_argument("--seed", type=int, default=None)

    # Fallback defaults (overridden by settings.json when present)
    parser.add_argument("--arch", type=str, default="resnet50")
    parser.add_argument("--dataset", type=str, default="Stanford_dog")
    parser.add_argument("--iid", action="store_true")
    parser.add_argument("--num_users", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--use_bbox", action="store_true")
    parser.add_argument("--patch_num", type=int, default=3)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--tr_frac", type=float, default=0.8)

    parser.add_argument("--gpuid", type=str, default="0")

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    saved_cfg = _read_settings(args.model_dir)
    data_args = _build_data_args(args, saved_cfg)

    model_path = _resolve_model_path(args)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model path not found: {model_path}")

    output_root = args.output_dir
    if output_root is None:
        if args.image_path is not None:
            sample_name = f"image_{os.path.splitext(os.path.basename(args.image_path))[0]}"
        elif args.use_global_dataset:
            sample_name = f"global_sample{args.sample_idx}"
        else:
            sample_name = f"client{args.client_idx}_sample{args.sample_idx}"
        output_root = os.path.join(args.model_dir, "local_analysis", sample_name)
    makedir(output_root)

    log, logclose = create_logger(log_filename=os.path.join(output_root, "local_analysis.log"))

    log(f"Using model: {model_path}")
    log(f"Reconstructed seed: {data_args.seed}")
    log(f"Distribution: iid={data_args.iid}, num_users={data_args.num_users}, alpha={data_args.alpha}")

    model = _build_model(data_args, model_path, device)

    last_layer_weight = model.last_layer.weight.detach().cpu()

    if args.image_path is not None:
        data_source = "image_path"
        image_batch, label = _load_from_image_path(data_args, args.image_path, args.image_label)
        log(f"Loaded image path sample: {args.image_path}, label={label}")
    elif args.use_global_dataset:
        data_source = "global_test"
        image_batch, label = _load_from_global_test_split(data_args, args.sample_idx)
        log(f"Loaded global test sample: sample_idx={args.sample_idx}, label={label}")
    else:
        data_source = "client_test"
        image_batch, label = _load_from_fl_split(data_args, args.client_idx, args.sample_idx)
        log(f"Loaded client test sample: client_idx={args.client_idx}, sample_idx={args.sample_idx}, label={label}")

    image_batch = image_batch.to(device)

    with torch.no_grad():
        # Same final-logit path as train_and_test_lr_schedule/local_test_global_model:
        # output, cosine_max_scores, project_max_distances, project_distances = model(image)
        output, cosine_max_scores, project_max_distances, project_distances = model(image_batch)
        conv_output, _ = model.push_forward(image_batch)

    final_logits = output[0]
    score_vec = project_max_distances[0]
    pred_cls = int(torch.argmax(final_logits).item())

    k_cls = min(args.topk_classes, int(final_logits.shape[0]))
    topk_logits, topk_classes = torch.topk(final_logits, k=k_cls)

    gt_label_name = _label_to_class_name(data_args, label)

    log(f"Predicted class: {pred_cls}")
    log(f"Ground truth label: {label}")
    if gt_label_name is not None:
        log(f"Ground truth label name: {gt_label_name}")
    log(f"Prediction correct: {pred_cls == label}")

    np.save(os.path.join(output_root, "final_logits.npy"), final_logits.detach().cpu().numpy())
    np.save(os.path.join(output_root, "score_vector_project_max_distances.npy"), score_vec.detach().cpu().numpy())

    for i in range(k_cls):
        cls_i = int(topk_classes[i].item())
        logit_i = float(topk_logits[i].item())
        score_i = float(score_vec[cls_i].item())
        diag_weight_i = float(last_layer_weight[cls_i, cls_i].item())
        diag_contribution_i = diag_weight_i * score_i
        log(
            f"Top-{i+1} class={cls_i}, final_logit={logit_i:.6f}, prototype_score={score_i:.6f}, "
            f"diag_last_layer_weight={diag_weight_i:.6f}, diag_contribution={diag_contribution_i:.6f}"
        )

    original_img = _save_preprocessed_img(
        os.path.join(output_root, "original_img.png"),
        image_batch.detach().cpu(),
        index=0,
    )

    feat_h = int(conv_output.shape[2])
    feat_w = int(conv_output.shape[3])
    pred_class_patch_scores = project_distances[0, pred_cls]  # shape: (H*W)

    # Save TesNet-style class activation heatmaps from projection scores.
    act_dir = os.path.join(output_root, "class_activation_maps")
    makedir(act_dir)
    img_h, img_w = original_img.shape[0], original_img.shape[1]

    pred_act_2d = pred_class_patch_scores.detach().cpu().numpy().reshape(feat_h, feat_w)
    _save_activation_overlay(
        os.path.join(act_dir, f"predicted_class_{pred_cls}"),
        pred_act_2d,
        original_img,
        img_h,
        img_w,
    )

    for i in range(k_cls):
        cls_i = int(topk_classes[i].item())
        cls_act_2d = project_distances[0, cls_i].detach().cpu().numpy().reshape(feat_h, feat_w)
        _save_activation_overlay(
            os.path.join(act_dir, f"top{i+1}_class_{cls_i}"),
            cls_act_2d,
            original_img,
            img_h,
            img_w,
        )

    img_h, img_w = original_img.shape[0], original_img.shape[1]
    top_class_patch_summary = []
    top_patches_predicted_class = []
    for i in range(k_cls):
        cls_i = int(topk_classes[i].item())
        class_rank = i + 1
        cls_patch_scores = project_distances[0, cls_i]
        class_dir = os.path.join(output_root, "top_class_patch_overlays", f"top{class_rank}_class_{cls_i}")
        makedir(class_dir)

        patch_summary = _save_top_patch_artifacts(
            class_dir=class_dir,
            class_rank=class_rank,
            class_idx=cls_i,
            patch_scores=cls_patch_scores,
            feat_h=feat_h,
            feat_w=feat_w,
            img_h=img_h,
            img_w=img_w,
            original_img=original_img,
            topk_patches=args.topk_patches,
            log=log,
        )

        top_class_patch_summary.append(
            {
                "rank": class_rank,
                "class": cls_i,
                "top_patches": patch_summary,
            }
        )
        if cls_i == pred_cls:
            top_patches_predicted_class = patch_summary

    summary = {
        "model_path": model_path,
        "seed": int(data_args.seed),
        "data_source": data_source,
        "client_idx": None if data_source != "client_test" else int(args.client_idx),
        "sample_idx": int(args.sample_idx),
        "predicted_class": pred_cls,
        "ground_truth_label": int(label),
        "ground_truth_label_name": gt_label_name,
        "prediction_correct": bool(pred_cls == label),
        "top_classes": [
            {
                "rank": i + 1,
                "class": int(topk_classes[i].item()),
                "final_logit": float(topk_logits[i].item()),
                "prototype_score": float(score_vec[int(topk_classes[i].item())].item()),
            }
            for i in range(k_cls)
        ],
        "top_patches_predicted_class": top_patches_predicted_class,
        "top_patches_by_top_class": top_class_patch_summary,
    }

    with open(os.path.join(output_root, "analysis_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    log("Saved: final_logits.npy, score_vector_project_max_distances.npy, class activation map overlays, per-top-class patch overlays, and analysis_summary.json")
    logclose()


if __name__ == "__main__":
    main()
