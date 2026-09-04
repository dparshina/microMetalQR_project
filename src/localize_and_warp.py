from __future__ import annotations
import warnings

import cv2
import numpy as np

warnings.filterwarnings("ignore")


def estimate_module_size(gray, default=10.0):
    height, width = gray.shape
    estimates = []
    for frac in [0.3, 0.4, 0.5, 0.6, 0.7]:
        for axis in (0, 1):
            line = gray[int(height * frac), :] if axis == 0 else gray[:, int(width * frac)]
            line = np.convolve(line.astype(np.float32), np.ones(3) / 3, mode="same")
            transitions = np.where(np.diff((line < line.mean()).astype(int)))[0]
            if len(transitions) < 10:
                continue
            small_gap = np.percentile(np.diff(transitions), 25)
            if 2 < small_gap < 50:
                estimates.append(small_gap)

    if len(estimates) < 3:
        return default
    estimates = np.array(estimates)
    if estimates.std() / max(estimates.mean(), 1e-6) > 0.3:
        return default
    return float(np.median(estimates))


def build_localization_binary(gray):
    sigma = max(gray.shape[1] // 10, 20)
    background = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), sigmaX=sigma)
    background = np.clip(background, 1, None)
    normalized = np.clip((gray.astype(np.float32) / background) * 128, 0, 255).astype(np.uint8)

    
    denoised = cv2.fastNlMeansDenoising(normalized, h=12,
                                        templateWindowSize=7, searchWindowSize=21)
    denoised = cv2.bilateralFilter(denoised, d=5, sigmaColor=40, sigmaSpace=40)

    
    ms = estimate_module_size(denoised)
    median_ksize = max(3, int(ms * 0.15)) | 1
    smoothed = cv2.medianBlur(denoised, median_ksize)

    
    _, thresholded = cv2.threshold(smoothed, 0, 255,
                                   cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    close_ksize = max(3, int(ms * 0.2))
    binary = cv2.morphologyEx(
        thresholded, cv2.MORPH_CLOSE, np.ones((close_ksize, close_ksize), np.uint8))

    return binary, ms


def try_cv2_detector(binary, gray):
    detector = cv2.QRCodeDetector()
    ok, corners = detector.detect(binary)
    if ok and corners is not None and len(corners) > 0:
        return corners[0]
    ok, corners = detector.detect(gray)
    if ok and corners is not None and len(corners) > 0:
        return corners[0]
    return None


def try_wechat_detector(gray):
    try:
        detector = cv2.wechat_qrcode_WeChatQRCode()
    except (AttributeError, cv2.error):
        return None
    try:
        _, corners = detector.detectAndDecode(gray)
        if corners is not None and len(corners) > 0:
            return np.array(corners[0], dtype="float32")
    except cv2.error:
        return None
    return None


def find_finder_patterns_manual(binary, ms):

    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE,
                                            cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return None
    height, width = binary.shape
    image_area = height * width


    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 0.003 * image_area or area > 0.10 * image_area:
            continue
        x, y, box_w, box_h = cv2.boundingRect(contour)
        if abs(box_w - box_h) > 0.35 * max(box_w, box_h):
            continue
        center_y, center_x = y + box_h / 2.0, x + box_w / 2.0
        candidates.append((center_y, center_x, area))
    if len(candidates) < 3:
        return None


    candidates.sort(key=lambda t: -t[2])
    merged = [False] * len(candidates)
    groups = []
    for i, cand_i in enumerate(candidates):
        if merged[i]: continue
        group = [cand_i]; merged[i] = True
        for j in range(i + 1, len(candidates)):
            if merged[j]: continue
            cand_j = candidates[j]
            if abs(cand_i[0] - cand_j[0]) < ms * 2 and abs(cand_i[1] - cand_j[1]) < ms * 2:
                group.append(cand_j); merged[j] = True
        group_ys = [c[0] for c in group]; group_xs = [c[1] for c in group]
        groups.append((np.mean(group_ys), np.mean(group_xs), max(c[2] for c in group)))
    if len(groups) < 3:
        return None

    groups.sort(key=lambda t: -t[2])
    centers = np.array([(g[0], g[1]) for g in groups[:3]])
    tl_idx = int(np.argmin(centers[:, 0] + centers[:, 1]))   
    tr_idx = int(np.argmax(centers[:, 1] - centers[:, 0]))   
    bl_idx = int(np.argmax(centers[:, 0] - centers[:, 1]))   
    if len({tl_idx, tr_idx, bl_idx}) != 3:
        return None
    top_left = centers[tl_idx]; top_right = centers[tr_idx]; bottom_left = centers[bl_idx]


    bottom_right = top_right + (bottom_left - top_left)

    corners = np.array([[top_left[1], top_left[0]],
                        [top_right[1], top_right[0]],
                        [bottom_right[1], bottom_right[0]],
                        [bottom_left[1], bottom_left[0]]], dtype="float32")

    offset = ms * 3.5
    corners[0] += [-offset, -offset]  
    corners[1] += [+offset, -offset]  
    corners[2] += [+offset, +offset]  
    corners[3] += [-offset, +offset]  
    return corners


def localize_qr(gray):
    binary, ms = build_localization_binary(gray)

    corners = try_cv2_detector(binary, gray)
    if corners is not None and is_quad_sane(corners, gray.shape):
        return corners, 'cv2', ms

    corners = try_wechat_detector(gray)
    if corners is not None and is_quad_sane(corners, gray.shape):
        return corners, 'wechat', ms

    corners = find_finder_patterns_manual(binary, ms)
    if corners is not None and is_quad_sane(corners, gray.shape):
        return corners, 'manual', ms

    return None, 'fail', ms


def is_quad_sane(corners, shape, tol=0.5):
    height, width = shape
    corners = np.asarray(corners).reshape(-1, 2)
    if len(corners) < 4:
        return False

    slack = max(height, width) * 0.05
    if (corners[:, 0] < -slack).any() or (corners[:, 0] > width + slack).any():
        return False
    if (corners[:, 1] < -slack).any() or (corners[:, 1] > height + slack).any():
        return False

    top_edge = np.linalg.norm(corners[1] - corners[0])
    left_edge = np.linalg.norm(corners[3] - corners[0])
    if max(top_edge, left_edge) / max(min(top_edge, left_edge), 1) > 1 + tol:
        return False
    return True


def estimate_module_size_heavy(gray, default=8.0):
    height, width = gray.shape
    estimates = []
    for frac in [0.35, 0.45, 0.5, 0.55, 0.65]:
        for axis in (0, 1):
            line = gray[int(height * frac), :] if axis == 0 else gray[:, int(width * frac)]
            line = np.convolve(line.astype(np.float32), np.ones(3) / 3, mode="same")
            transitions = np.where(np.diff((line < line.mean()).astype(int)))[0]
            if len(transitions) < 10: continue
            small_gap = np.percentile(np.diff(transitions), 25)
            if 2 < small_gap < 40: estimates.append(small_gap)
    if len(estimates) < 3: return default
    estimates = np.array(estimates)
    if estimates.std() / max(estimates.mean(), 1e-6) > 0.3: return default
    return float(np.median(estimates))


def remove_small_components(binary, module_px):
    min_area = max(4, int((module_px * 0.3) ** 2))
    kept = np.zeros_like(binary)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    for label in range(1, n_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            kept[labels == label] = 255
    return kept


def process_metal_heavy(gray):
    background = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), sigmaX=gray.shape[1] // 6)
    background = np.clip(background, 1, None)
    normalized = np.clip((gray.astype(np.float32) / background) * 128, 0, 255).astype(np.uint8)

    denoised = cv2.fastNlMeansDenoising(normalized, h=30,
                                        templateWindowSize=7, searchWindowSize=35)
    bilateral = cv2.bilateralFilter(denoised, d=7, sigmaColor=50, sigmaSpace=50)

    lo, hi = np.percentile(bilateral, 5), np.percentile(bilateral, 95)
    stretched = np.clip((bilateral.astype(np.float32) - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    module_px = estimate_module_size_heavy(stretched)
    median_ksize = max(3, int(module_px * 0.4)) | 1
    smoothed = cv2.medianBlur(stretched, median_ksize)

    otsu_threshold, _ = cv2.threshold(smoothed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, binary = cv2.threshold(smoothed, otsu_threshold * 0.85, 255, cv2.THRESH_BINARY_INV)
    open_ksize = max(2, int(module_px * 0.3))
    close_ksize = max(2, int(module_px * 0.2))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((open_ksize, open_ksize), np.uint8))
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, np.ones((close_ksize, close_ksize), np.uint8))
    cleaned = remove_small_components(cleaned, module_px)
    return cleaned, module_px


def primary_pipeline(image_path):
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None: return None, 'cannot_read', 0

    binary, ms = process_metal_heavy(gray)
    inverted = 255 - binary
    detector = cv2.QRCodeDetector()
    ok, corners = detector.detect(inverted)
    method = 'cv2'
    if not ok or corners is None:
        equalized = cv2.equalizeHist(inverted)
        ok, corners = detector.detect(equalized)
        method = 'cv2_eq'
    if not ok or corners is None:
        return None, 'fail', ms
    corners = np.asarray(corners).reshape(-1, 2)
    if not is_quad_sane(corners, binary.shape):
        return None, 'fail_quad', ms

    raw_color = cv2.imread(str(image_path))
    orig_h, orig_w = raw_color.shape[:2]
    binary_h, binary_w = binary.shape
    scale = np.array([orig_w / binary_w, orig_h / binary_h], dtype="float32")
    src_corners = corners.astype("float32") * scale

    w_out = int(max(np.linalg.norm(src_corners[2] - src_corners[3]),
                    np.linalg.norm(src_corners[1] - src_corners[0])))
    h_out = int(max(np.linalg.norm(src_corners[1] - src_corners[2]),
                    np.linalg.norm(src_corners[0] - src_corners[3])))
    out_size = max(w_out, h_out)
    if out_size < 10: return None, 'fail_degenerate', ms

    dst = np.array([[0, 0], [out_size-1, 0], [out_size-1, out_size-1], [0, out_size-1]], dtype="float32")
    transform = cv2.getPerspectiveTransform(src_corners, dst)
    warped = cv2.warpPerspective(cv2.cvtColor(raw_color, cv2.COLOR_BGR2GRAY), transform, (out_size, out_size))
    return warped, method, ms


def fallback_pipeline(image_path):
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None: return None, 'cannot_read', 0

    corners, method, ms = localize_qr(gray)
    if corners is None: return None, 'fail', ms
    corners = np.asarray(corners, dtype="float32").reshape(-1, 2)
    top_left, top_right, bottom_right, bottom_left = corners[0], corners[1], corners[2], corners[3]

    w_out = int(max(np.linalg.norm(bottom_right - bottom_left), np.linalg.norm(top_right - top_left)))
    h_out = int(max(np.linalg.norm(top_right - bottom_right), np.linalg.norm(top_left - bottom_left)))
    out_size = max(w_out, h_out)
    if out_size < 10: return None, 'fail_degenerate', ms

    dst = np.array([[0, 0], [out_size-1, 0], [out_size-1, out_size-1], [0, out_size-1]], dtype="float32")
    src = np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32")
    transform = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(gray, transform, (out_size, out_size))
    return warped, method, ms
