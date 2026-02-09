"""
02_region_proposal.py

Session 2 (structure-based):
Propose candidate parking regions by detecting stall-like line structure
(parallel short lines) rather than color/brightness.

Outputs:
- outputs/02_scoremap_<name>.png   (visual score heatmap)
- outputs/02_mask_<name>.png       (binary candidate regions)
- outputs/02_candidates_<name>.png (outlined regions over original image)

Usage:
    python 02_region_proposal.py --image "path/to/img.png" --name "example" --mask "path/to/property_mask.png"
"""

import os
import argparse
import cv2
import numpy as np


def angle_dist(a, b):
    # distance on [0,pi) with wrap-around
    d = abs(a - b)
    return min(d, np.pi - d)

def alignment_weight(peak_angle, dominant_angles, tol=np.deg2rad(12)):
    if not dominant_angles:
        return 1.0
    d = min(angle_dist(peak_angle, d0) for d0 in dominant_angles)
    # inside tol -> 1.0, outside tol -> decay
    if d <= tol:
        return 1.0
    return float(np.exp(-(d - tol) / np.deg2rad(10)))  # smooth decay


def dominant_from_seed_windows(gray, prop_mask, window, stride, seed_mask, topk=2):
    angles = []
    h, w = gray.shape

    for y in range(0, h - window + 1, stride):
        for x in range(0, w - window + 1, stride):
            # only consider windows that are "seed-positive"
            if seed_mask[y:y+window, x:x+window].mean() < 10:
                continue
            roi = gray[y:y+window, x:x+window]
            edges = cv2.Canny(roi, 60, 180)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=35, minLineLength=18, maxLineGap=6)
            if lines is None:
                continue
            for x1,y1,x2,y2 in lines[:,0]:
                ang = np.arctan2(y2-y1, x2-x1) % np.pi
                angles.append(ang)

    if len(angles) < 30:
        return []

    angles = np.array(angles, dtype=np.float32)
    hist, bin_edges = np.histogram(angles, bins=36, range=(0, np.pi))
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    idxs = np.argsort(hist)[-topk:]
    return centers[idxs].tolist()


def dominant_orientations(gray, prop_mask, topk=2):
    edges = cv2.Canny(gray, 60, 180)
    edges = cv2.bitwise_and(edges, edges, mask=prop_mask)

    lines = cv2.HoughLines(edges, rho=1, theta=np.pi/180, threshold=180)
    if lines is None:
        return []

    angles = np.array([(l[0][1] % np.pi) for l in lines], dtype=np.float32)

    hist, bin_edges = np.histogram(angles, bins=36, range=(0, np.pi))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    idxs = np.argsort(hist)[-topk:]          # top bins
    dominant = bin_centers[idxs]             # centers, not edges

    return dominant.tolist()

def aligned(angle, dominant_angles, tol=np.deg2rad(10)):
    return any(abs(angle - d) < tol for d in dominant_angles)

# -----------------------------
# Scoring: "parking-like striping"
# -----------------------------

def line_parking_score(gray_roi, dominant_angles):
    """
    Score a small ROI for stall-striping structure.

    Returns:
        score in [0,1] roughly (not guaranteed but typically).
    """
    edges = cv2.Canny(gray_roi, 60, 180)

    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=35,
        minLineLength=18,
        maxLineGap=6
    )

    if lines is None:
        return 0.0

    angles = []
    lengths = []

    for x1, y1, x2, y2 in lines[:, 0]:
        dx = x2 - x1
        dy = y2 - y1
        L = float(np.hypot(dx, dy))
        if L < 18:
            continue

        ang = np.arctan2(dy, dx) % np.pi  # fold to [0, pi)
        angles.append(ang)
        lengths.append(L)

    if len(angles) < 10:
        return 0.0

    angles = np.array(angles, dtype=np.float32)
    lengths = np.array(lengths, dtype=np.float32)

    # Orientation concentration: strong single peak = parking rows/stripes
    hist, _ = np.histogram(angles, bins=36, range=(0, np.pi))
    strength = float(hist.max() / (hist.sum() + 1e-6))


    #if roi not aligned with property, return 0
    peak_bin = np.argmax(hist)
    peak_angle = (peak_bin / 36.0) * np.pi

    if not aligned(peak_angle, dominant_angles):
        return 0.0

    # Prefer short-ish segments typical of stall dividers (not huge long edges)
    med_len = float(np.median(lengths))
    len_score = float(np.clip(med_len / 40.0, 0.0, 1.0))

    n = len(lengths)

    # reject if dominated by long segments (often building/road edges)
    med_len = float(np.median(lengths))
    if med_len > 70:
        return 0.0

    count_score = float(np.clip(n / 40.0, 0.0, 1.0))

    # existing strength + len_score
    w_align = alignment_weight(peak_angle, dominant_angles)
    score = w_align * (0.65 * strength + 0.15 * len_score + 0.20 * count_score)
    return float(np.clip(score, 0.0, 1.0))


# -----------------------------
# Sliding window score map
# -----------------------------

def compute_score_map(gray, property_mask, dominant_angles, window=256, stride=96):
    h, w = gray.shape
    score_map = np.zeros((h, w), dtype=np.float32)

    m = (property_mask > 0).astype(np.uint8)
    integral = cv2.integral(m)

    def mask_coverage(x, y, win):
        x2 = min(x + win, w)
        y2 = min(y + win, h)
        s = integral[y2, x2] - integral[y, x2] - integral[y2, x] + integral[y, x]
        area = (x2 - x) * (y2 - y)
        return 0.0 if area == 0 else float(s) / float(area)

    for y in range(0, h - window + 1, stride):
        for x in range(0, w - window + 1, stride):
            if mask_coverage(x, y, window) < 0.6:
                continue

            roi = gray[y:y + window, x:x + window]
            score = line_parking_score(roi, dominant_angles)

            score_map[y:y + window, x:x + window] = np.maximum(
                score_map[y:y + window, x:x + window],
                score
            )

    return score_map

# -----------------------------
# Threshold score map -> candidate mask
# -----------------------------

def score_map_to_mask(score_map, property_mask, thresh=0.20):
    """
    Convert score map to binary candidate mask.
    """
    candidate = (score_map >= thresh).astype(np.uint8) * 255
    candidate = cv2.bitwise_and(candidate, candidate, mask=property_mask)

    # Clean up / merge nearby windows
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1, 41))  #TODO: rotated kernel
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel, iterations=2) #maybe drop
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, kernel, iterations=1)

    return candidate


# -----------------------------
# Outline regions for debug
# -----------------------------

def outline_regions(image_bgr, mask, min_area=10000):
    outlined = image_bgr.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    kept = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        kept.append(c)
        cv2.drawContours(outlined, [c], -1, (0, 255, 0), 3)

    return kept, outlined

def extract_region_objects(candidate_mask, score_map, min_area=10000):
    """
    Build region objects the rest of the pipeline can consume.

    Returns:
        regions: list[dict] sorted best-first
            {
              "bbox": (x, y, w, h),
              "area_px": int,
              "contour": np.ndarray,
              "score_mean": float,
              "score_max": float,
              "rotated_rect": ((cx,cy),(w,h),angle),
              "rotated_box_pts": np.ndarray((4,2), int),
              "angle_deg": float (normalized to [0,180))
            }
    """
    regions = []

    contours, _ = cv2.findContours(candidate_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(c)

        # Make a filled mask for this contour to compute score stats
        region_mask = np.zeros(candidate_mask.shape, dtype=np.uint8)
        cv2.drawContours(region_mask, [c], -1, 255, thickness=-1)

        scores = score_map[region_mask > 0]
        score_mean = float(scores.mean()) if scores.size else 0.0
        score_max = float(scores.max()) if scores.size else 0.0

        # Rotated bounding box (OpenCV minAreaRect)
        rect = cv2.minAreaRect(c)  # ((cx, cy), (bw, bh), angle)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # Normalize angle to [0, 180). OpenCV angle semantics:
        # - rect[2] in (-90, 0] when width < height, else in [0, 90)
        angle = float(rect[2])
        if rect[1][0] < rect[1][1]:
            angle = (angle + 90.0) % 180.0
        else:
            angle = angle % 180.0

        regions.append({
            "bbox": (int(x), int(y), int(w), int(h)),
            "area_px": int(area),
            "contour": c,
            "score_mean": score_mean,
            "score_max": score_max,
            "rotated_rect": rect,
            "rotated_box_pts": box,
            "angle_deg": angle,
        })

    # Best-first ordering: higher score then larger area
    regions.sort(key=lambda r: (r["score_mean"], r["area_px"]), reverse=True)
    return regions


# -----------------------------
# Main
# -----------------------------

def parking_lot_proposal(
    image,
    prop_mask,
    name,
    window=256,
    stride=96,
    thresh=0.20,
    debug=False,
):
    """

    Args:
        image: BGR image (np.uint8)
        prop_mask: binary property mask (same size as image)
        name: string identifier for outputs
        window: sliding window size (px)
        stride: sliding stride (px)
        thresh: parking score threshold
        debug: write debug images if True

    Returns:
        candidate_mask: uint8 binary mask
        regions: list of region dicts (for Session 3)
        score_map: float32 score map (optional downstream use)
    """
    if debug:
        print("parking lot proposal called on "+name)

    os.makedirs("outputs", exist_ok=True)

    # Ensure property mask is binary uint8
    prop_mask = (prop_mask > 0).astype(np.uint8) * 255

    # Convert to grayscale once
    if image.ndim>2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray=image

    #compute dominant orientations
    #dominant_angles = dominant_orientations(gray, prop_mask, topk=2)



    # --- Compute parking-likeness score map ---
    # PASS 1: compute score map with NO dominant angles
    score_map_1 = compute_score_map(gray, prop_mask, dominant_angles=[], window=window, stride=stride)

    # seeds = top X% of scores (inside property)
    scores = score_map_1[prop_mask > 0]
    t_seed = float(np.percentile(scores, 93)) if scores.size else thresh
    seed_mask = ((score_map_1 >= t_seed).astype(np.uint8) * 255)
    seed_mask = cv2.bitwise_and(seed_mask, seed_mask, mask=prop_mask)

    # estimate dominant angles from seed windows
    dominant_angles = dominant_from_seed_windows(gray, prop_mask, window, stride, seed_mask, topk=2)

    # PASS 2: recompute score map using soft alignment penalty
    score_map = compute_score_map(gray, prop_mask, dominant_angles, window=window, stride=stride)

    # --- Build candidate mask via seed + grow (THIS IS THE OUTPUT MASK) ---
    prop_scores = score_map[prop_mask > 0]
    if prop_scores.size == 0:
        candidate_mask = np.zeros_like(prop_mask)
    else:
        # Seeds: very confident
        t_hi = float(np.percentile(prop_scores, 93))
        seed = ((score_map >= t_hi).astype(np.uint8) * 255)

        # Grow: include medium-confidence neighboring windows
        t_lo = float(np.percentile(prop_scores, 80))
        cand = ((score_map >= t_lo).astype(np.uint8) * 255)

        seed = cv2.bitwise_and(seed, seed, mask=prop_mask)
        cand = cv2.bitwise_and(cand, cand, mask=prop_mask)

        # If seed is empty, fallback to just using cand at a higher threshold
        if seed.sum() == 0:
            t_fallback = float(np.percentile(prop_scores, 90))
            candidate_mask = ((score_map >= t_fallback).astype(np.uint8) * 255)
            candidate_mask = cv2.bitwise_and(candidate_mask, candidate_mask, mask=prop_mask)
        else:
            # morphological geodesic dilation approximation
            kernel = np.ones((21, 21), np.uint8)
            grown = seed.copy()
            for _ in range(3):
                grown = cv2.dilate(grown, kernel, iterations=1)
                grown = cv2.bitwise_and(grown, cand)

            candidate_mask = grown

    # light cleanup (optional)
    kernel = np.ones((15, 15), np.uint8)
    candidate_mask = cv2.morphologyEx(candidate_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    candidate_mask = cv2.morphologyEx(candidate_mask, cv2.MORPH_OPEN,  kernel, iterations=1)

    # --- Extract region objects for downstream pipeline ---
    regions = extract_region_objects(
        candidate_mask,
        score_map,
        min_area=10_000,
    )

    # --- Debug outputs ---
    if debug:
        score_vis = (np.clip(score_map, 0, 1) * 255).astype(np.uint8)
        cv2.imwrite(f"outputs/02_scoremap_{name}.png", score_vis)
        cv2.imwrite(f"outputs/02_mask_{name}.png", candidate_mask)

        outlined = image.copy()
        for r in regions:
            if "rotated_box_pts" in r:
                cv2.drawContours(outlined, [r["rotated_box_pts"]], -1, (0, 255, 0), 3)
            else:
                cv2.drawContours(outlined, [r["contour"]], -1, (0, 255, 0), 3)

        cv2.imwrite(f"outputs/02_candidates_{name}.png", outlined)

        print(f"[Session 2] Candidate regions: {len(regions)}")
        print(f"  outputs/02_scoremap_{name}.png")
        print(f"  outputs/02_mask_{name}.png")
        print(f"  outputs/02_candidates_{name}.png")

    return candidate_mask, regions, score_map

    

if __name__ == "__main__":
    parking_lot_proposal()
