import os
import json
import math
import torch
import argparse
import numpy as np
import gc
from tqdm import tqdm
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from PIL import Image

# GroundingDINO imports
from groundingdino.util.inference import load_model, predict, load_image

def distance_threshold_func(boxes): # list of [xc,yc,w,h]
    if len(boxes) == 0:
        return 0.0
    # find median index of boxes areas
    areas = [box[2]*box[3] for box in boxes]
    median_idx = np.argsort(areas)[len(areas)//2]
    median_box = boxes[median_idx]
    w = median_box[2]
    h = median_box[3]

    threshold = np.sqrt(w**2 + h**2) / 2.0
    
    return threshold

def calc_loc_metric(pred_boxes, gt_points): # list of [xc,yc,w,h], tensor of (nt,2)
    if len(pred_boxes) == 0:
        FN = len(gt_points)
        return 0, 0, FN, 0, 0, 0
    
    dist_threshold = distance_threshold_func(pred_boxes)
    pred_points = np.array([[box[0], box[1]] for box in pred_boxes])
    gt_points = np.array(gt_points)

    # create a cost matrix
    cost_matrix = cdist(pred_points, gt_points, metric='euclidean')
    
    # use Hungarian algorithm to find optimal assignment
    pred_indices, gt_indices = linear_sum_assignment(cost_matrix)
    
    # determine TP, FP, FN
    TP = 0
    for pred_idx, gt_idx in zip(pred_indices, gt_indices):
        if cost_matrix[pred_idx, gt_idx] < dist_threshold:
            TP += 1
    
    FP = len(pred_points) - TP
    FN = len(gt_points) - TP

    Precision = TP / (TP + FP) if TP + FP != 0 else 0.0
    Recall = TP / (TP + FN) if TP + FN != 0 else 0.0
    F1 = 2 * (Precision * Recall) / (Precision + Recall) if Precision + Recall != 0 else 0.0
    
    return TP, FP, FN, Precision, Recall, F1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./groundingdino/dataset/REC-8K/")
    parser.add_argument("--output_dir", type=str, default="./logs/REC-8K")
    parser.add_argument("--test-split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--config", type=str, default="./groundingdino/config/GroundingDINO_SwinT_OGC.py")
    parser.add_argument("--ckpt", type=str, default="./weights/groundingdino_swint_ogc.pth")
    parser.add_argument("--box_threshold", type=float, default=0.30)
    parser.add_argument("--text_threshold", type=float, default=0.25)
    parser.add_argument("-d", "--device", type=str, default='cuda:0')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logf = open(os.path.join(args.output_dir, f"log-{args.test_split}.txt"), "w")

    with open(os.path.join(args.data_path, "annotations.json")) as f:
        annotations = json.load(f)
    with open(os.path.join(args.data_path, "RE-splits.json")) as f:
        #splits = json.load(f)
        records = json.load(f)
    #samples = splits[args.test_split]
    samples = [r for r in records if r["splits"] == args.test_split]

    model = load_model(args.config, args.ckpt)
    model.eval()
    device = torch.device(args.device)
    model.to(device)

    MAE = 0.0
    RMSE = 0.0
    tot_TP = tot_FP = tot_FN = 0

    #for i, (im_id, ref_expr) in enumerate(tqdm(samples,bar_format="{n}/{total} [{elapsed}<{remaining}, {rate_fmt}]")):
    #    im_path = os.path.join(args.data_path, "images", im_id)
    for i, sample in enumerate(tqdm(samples, bar_format="{n}/{total} [{elapsed}<{remaining}, {rate_fmt}]")):
        im_id = sample["image"]
        ref_expr = sample["referring expression"]
        subject = sample.get("subject", "")
        visual_pattern = sample.get("visual pattern", "")
        location = sample.get("location", "")
        im_path = os.path.join(args.data_path, "images", im_id)

        # inference image
        image_source, img_tensor = load_image(im_path)
        img_tensor = img_tensor.to(device)

        img_pil = Image.open(im_path).convert("RGB")
        orig_w, orig_h = img_pil.size

        gt_pts = np.array(annotations[im_id][ref_expr]["points"])

        # predict normalized boxes
        boxes_norm, logits, phrases = predict(
            model=model,
            image=img_tensor,
            caption=ref_expr,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            remove_combined=True
        )

        boxes_arr = np.array(boxes_norm) * np.array([orig_w, orig_h, orig_w, orig_h])
        pred_boxes = [
            (float(cx), float(cy), float(w), float(h))
            for cx, cy, w, h in boxes_arr
        ]

        # count and evaluate
        gt_cnt = len(gt_pts)
        pred_cnt = len(pred_boxes)
        err = abs(pred_cnt - gt_cnt)
        MAE += err
        RMSE += err * err

        TP, FP, FN, P, R, F1 = calc_loc_metric(pred_boxes, gt_pts)

        tot_TP += TP
        tot_FP += FP
        tot_FN += FN
        print(f"pred_cnt={pred_cnt:4d}, gt_cnt={gt_cnt:4d}, error={abs(pred_cnt-gt_cnt):4d}")
        logf.write(f"{im_id},{ref_expr},{pred_cnt},{gt_cnt},{err},{TP},{FP},{FN},{P:.4f},{R:.4f},{F1:.4f}\n")
        logf.flush()
        torch.cuda.empty_cache()
        gc.collect()

    N = len(samples)
    MAE /= N
    RMSE = math.sqrt(RMSE / N)
    Prec = tot_TP / (tot_TP + tot_FP) if tot_TP + tot_FP > 0 else 0.0
    Rec = tot_TP / (tot_TP + tot_FN) if tot_TP + tot_FN > 0 else 0.0
    F1g = 2 * Prec * Rec / (Prec + Rec) if Prec + Rec > 0 else 0.0

    print(f"MAE: {MAE:.2f}, RMSE: {RMSE:.2f}")
    print(f"Precision: {Prec:.4f}, Recall: {Rec:.4f}, F1: {F1g:.4f}")
    logf.write(f"MAE: {MAE:.2f}, RMSE: {RMSE:.2f}\n")
    logf.write(f"Precision: {Prec:.4f}, Recall: {Rec:.4f}, F1: {F1g:.4f}\n")
    logf.close()
