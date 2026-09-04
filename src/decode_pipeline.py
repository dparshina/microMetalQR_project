from __future__ import annotations

import warnings

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage import filters, restoration, feature, exposure, morphology, measure

warnings.filterwarnings("ignore")

QR_SIZE = 33
TARGET = 64  


BLOB_CENTER_REL = (0.4, 0.4)
BLOB_HALF_SIDE_REL = 0.2 


def load_gray(path) -> np.ndarray:
    return np.array(Image.open(path).convert("L"), dtype=np.float32)


def estimate_module_grid(gray: np.ndarray, qr_n: int = QR_SIZE):
    height, width = gray.shape
    module_size = min(height, width) / qr_n
    grad_rows = np.abs(np.diff(gray, axis=0)).mean(axis=1)  
    grad_cols = np.abs(np.diff(gray, axis=1)).mean(axis=0)  

    def best_offset(grad_1d, module_size):
        length = len(grad_1d)
        best_off, best_score = 0.0, -np.inf
        for off in np.linspace(-module_size / 2, module_size / 2, 31):
            boundaries = off + np.arange(1, qr_n) * module_size
            boundaries = boundaries[(boundaries >= 1) & (boundaries < length - 1)].astype(int)
            if len(boundaries) < qr_n - 3:
                continue
            score = grad_1d[boundaries].sum()
            if score > best_score:
                best_score, best_off = score, off
        return float(np.clip(best_off, -module_size * 0.3, module_size * 0.3))

    return float(module_size), best_offset(grad_rows, module_size), best_offset(grad_cols, module_size)


def flatten_illumination(gray, ms):
    sigma = max(ms * 3.0, 8.0)
    background = np.clip(ndi.gaussian_filter(gray, sigma), 1.0, None)
    flat = gray / background
    lo, hi = np.percentile(flat, [1, 99])
    return (np.clip((flat - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.float32)


def denoise_grain(gray, ms):
    median = ndi.median_filter(gray, size=3)
    bilateral = restoration.denoise_bilateral(
        median / 255.0,
        sigma_color=0.08,
        sigma_spatial=max(ms / 8.0, 1.5),
        channel_axis=None,
    )
    bilateral = ndi.gaussian_filter(bilateral, sigma=max(ms / 15.0, 0.6))
    return (bilateral * 255).astype(np.float32)


def preprocess(gray):
    ms, oy, ox = estimate_module_grid(gray)
    flat = flatten_illumination(gray, ms)
    denoised = denoise_grain(flat, ms)
    return {"gray": gray, "flat": flat, "denoised": denoised, "ms": ms, "oy": oy, "ox": ox}


def extract_module(img, ms, oy, ox, r, c):
    y0 = int(round(oy + r * ms))
    y1 = int(round(oy + (r + 1) * ms))
    x0 = int(round(ox + c * ms))
    x1 = int(round(ox + (c + 1) * ms))
    return img[max(y0, 0):y1, max(x0, 0):x1]


def build_functional_mask(qr_n: int = QR_SIZE, quiet: int = 4):
    total = qr_n + 2 * quiet
    mask = np.zeros((total, total), dtype=bool)


    mask[:quiet, :] = True; mask[-quiet:, :] = True
    mask[:, :quiet] = True; mask[:, -quiet:] = True


    for i in range(7):
        for j in range(7):
            mask[quiet + i][quiet + j] = True
            mask[quiet + i][total - quiet - 7 + j] = True
            mask[total - quiet - 7 + i][quiet + j] = True


    for i in range(8):
        mask[quiet + i][quiet + 7] = True
        mask[quiet + 7][quiet + i] = True


    for i in range(8, total - 8):
        mask[quiet + 6][quiet + i] = True
        mask[quiet + i][quiet + 6] = True


    for i in range(9):
        mask[quiet + 8][quiet + i] = True
        mask[quiet + i][quiet + 8] = True

    return mask[quiet:quiet + qr_n, quiet:quiet + qr_n]


FUNCTIONAL = build_functional_mask()

def to_gray64(img):
    if img.ndim == 3:
        img_pil = Image.fromarray(img).convert("L")
    else:
        img_pil = Image.fromarray(img.astype(np.uint8))
    return np.array(img_pil.resize((TARGET, TARGET), Image.LANCZOS))


def local_contrast(img, r, c, radius):
    height, width = img.shape
    ys, xs = np.mgrid[0:height, 0:width]
    dist = np.sqrt((xs - c) ** 2 + (ys - r) ** 2)
    inner = (dist <= radius * 0.9)
    outer = (dist > radius * 1.2) & (dist <= radius * 2.2)
    inner_mean = float(img[inner].mean()) if inner.any() else 0.0
    outer_mean = float(img[outer].mean()) if outer.any() else 0.0
    return inner_mean - outer_mean


def is_isotropic(img, r, c, radius, n_dirs=12):
    height, width = img.shape
    angles = np.linspace(0, 2 * np.pi, n_dirs, endpoint=False)

    def ring(dist):
        return np.array([
            float(img[int(np.clip(r + dist * np.sin(a), 0, height - 1)),
                      int(np.clip(c + dist * np.cos(a), 0, width - 1))])
            for a in angles
        ])

    inner = ring(max(1.5, radius * 0.45))
    inner_mean = inner.mean()
    if inner_mean < 0.05 or inner.std() / (inner_mean + 1e-6) > 0.6:
        return False

    grad_y, grad_x = np.gradient(img)
    boundary_r = max(2.0, radius * 0.85)
    gradient_angles = []
    for a in angles:
        row = int(np.clip(r + boundary_r * np.sin(a), 0, height - 1))
        col = int(np.clip(c + boundary_r * np.cos(a), 0, width - 1))
        gradient_angles.append(float(np.arctan2(grad_y[row, col], grad_x[row, col])))
    coherence = abs(np.exp(1j * np.array(gradient_angles)).mean())
    if coherence > 0.72:
        return False

    outer = ring(max(3.0, radius * 1.6))
    if float((outer < inner_mean * 0.65).mean()) < 0.5:
        return False
    return True


def blob_log_score(log_img, raw_img):
    size = log_img.shape[0]
    expected_cy = BLOB_CENTER_REL[0] * size  
    expected_cx = BLOB_CENTER_REL[1] * size
    margin = size * 0.05
    try:
        blobs = feature.blob_log(
            log_img,
            min_sigma=2.0,
            max_sigma=12.0,
            num_sigma=10,
            threshold=0.030,
            overlap=0.5,
        )
    except Exception:
        return -3.0, None
    if len(blobs) == 0:
        return -4.0, None

    best_score, best_blob = -10.0, None
    for r, c, sigma in blobs:
        radius = sigma * 1.4142
        if radius < 5.0 or radius > size * 0.35:
            continue
        if r < margin or r > size - margin or c < margin or c > size - margin:
            continue
        if not is_isotropic(raw_img, r, c, radius):
            continue

        dist = np.sqrt((r - expected_cy) ** 2 + (c - expected_cx) ** 2) / (size * 0.5 * 1.4142)
        contrast = local_contrast(log_img, r, c, radius)

        score = 0.0
        if dist < 0.20:
            score += 3.5
        elif dist < 0.35:
            score += 2.5
        elif dist < 0.50:
            score += 1.0
        else:
            score -= 1.5
        if contrast > 0.20:
            score += 4.0
        elif contrast > 0.10:
            score += 2.5
        elif contrast > 0.05:
            score += 1.0
        else:
            score -= 1.5

        if 7.0 < radius < 16.0:
            score += 1.5
        else:
            score -= 0.5

        if score > best_score:
            best_score, best_blob = score, (float(r), float(c), float(radius))
    return (best_score if best_score > -10 else -4.0), best_blob


def compute_shape_score(smoothed, is_dark_bg):
    threshold_otsu = filters.threshold_otsu(smoothed)
    threshold_local = filters.threshold_local(smoothed, block_size=15, offset=0.02)
    scores = []
    for foreground in [smoothed > threshold_otsu, smoothed > threshold_local]:
        binary = foreground.astype(np.uint8) * 255
        if not is_dark_bg:
            binary = 255 - binary
        cleaned = morphology.opening(binary > 127, morphology.disk(2)).astype(np.uint8) * 255
        features = extract_features(cleaned)
        scores.append(score_shape_features(features))
    return float(np.mean(scores))


def extract_features(binary):
    height, width = binary.shape
    total = height * width
    features = {}
    white = binary > 127
    features["white_ratio"] = float(white.sum() / total)

    labeled, n_regions = measure.label(white, return_num=True, connectivity=2)
    if n_regions == 0:
        return zero_features(features)
    regions = sorted(measure.regionprops(labeled), key=lambda r: r.area, reverse=True)
    largest = regions[0]
    area = largest.area
    perimeter = largest.perimeter

    features["area_ratio"] = float(area / total)
    features["circularity"] = float(min(4 * np.pi * area / (perimeter ** 2 + 1e-6), 1.0))
    features["aspect_ratio"] = float(largest.major_axis_length / (largest.minor_axis_length + 1e-6))
    centroid_y, centroid_x = largest.centroid

    features["centroid_dist"] = float(
        np.sqrt((centroid_x - BLOB_CENTER_REL[1] * width) ** 2 + (centroid_y - BLOB_CENTER_REL[0] * height) ** 2)
        / (np.sqrt(height ** 2 + width ** 2) / 2.0)
    )
    features["solidity"] = float(area / (largest.convex_area + 1e-6))
    features["extent"] = float(area / (largest.bbox_area + 1e-6))
    features["eccentricity"] = float(largest.eccentricity)

    min_r, min_c, max_r, max_c = largest.bbox
    box_h = max_r - min_r
    box_w = max_c - min_c
    features["bbox_squareness"] = float(min(box_h, box_w) / (max(box_h, box_w) + 1e-6))
    border_margin = 3
    features["touches_border"] = float(
        min_r <= border_margin or min_c <= border_margin
        or max_r >= height - border_margin or max_c >= width - border_margin
    )
    features["n_large"] = float(sum(1 for region in regions if region.area > total * 0.05))
    return features


def zero_features(features):
    features.update({
        "area_ratio": 0, "circularity": 0, "aspect_ratio": 10,
        "centroid_dist": 1, "solidity": 0, "extent": 0, "eccentricity": 1,
        "bbox_squareness": 0, "touches_border": 1, "n_large": 0,
    })
    return features


def score_shape_features(features):
    area = features.get("area_ratio", 0)
    if area < 0.04 or area > 0.55:
        return -6.0
    if features.get("extent", 0) > 0.94:
        return -5.0
    if features.get("aspect_ratio", 10) > 4.0:
        return -5.0

    score = 0.0
    circularity = features.get("circularity", 0)
    score += 3.0 if circularity > 0.65 else (1.0 if circularity > 0.5 else (-0.5 if circularity > 0.35 else -2.5))
    solidity = features.get("solidity", 0)
    score += 1.5 if solidity > 0.8 else (0.5 if solidity > 0.65 else -1.0)
    aspect_ratio = features.get("aspect_ratio", 10)
    score += 1.5 if aspect_ratio < 1.35 else (0.5 if aspect_ratio < 1.7 else -1.5)
    eccentricity = features.get("eccentricity", 1)
    score += 1.0 if eccentricity < 0.5 else (-1.0 if eccentricity > 0.75 else 0.0)
    extent = features.get("extent", 0)
    if 0.6 < extent < 0.95:
        score += 1.0
    if features.get("touches_border", 1) == 0:
        score += 2.0
    else:
        score -= 3.0
    centroid_dist = features.get("centroid_dist", 1)
    score += 1.5 if centroid_dist < 0.30 else (0.5 if centroid_dist < 0.50 else -1.0)
    n_large = features.get("n_large", 0)
    score += 0.5 if n_large == 1 else (-1.0 if n_large == 0 else -0.5)
    if 0.08 < area < 0.45:
        score += 0.5
    return score


def detect_blob_square(img):
    gray = to_gray64(img)
    mean_intensity = float(gray.mean())
    is_dark_bg = mean_intensity < 128

    enhanced = exposure.equalize_adapthist(gray / 255.0, clip_limit=0.03)
    smoothed = filters.gaussian(enhanced, sigma=1.0)

    log_img = smoothed if is_dark_bg else (1.0 - smoothed)
    raw_norm = gray / 255.0 if is_dark_bg else (255.0 - gray) / 255.0
    raw_img = filters.gaussian(raw_norm, sigma=0.8)

    log_score, best_blob = blob_log_score(log_img, raw_img)
    shape_score = compute_shape_score(smoothed, is_dark_bg)


    blended_score = 0.4 * log_score + 0.6 * shape_score
    return {
        "has_blob": blended_score > 0,
        "score": float(blended_score),
        "log_score": float(log_score),
        "shape_score": float(shape_score),
        "bg": "dark" if is_dark_bg else "light",
        "blob": best_blob,
    }



def make_sbd():
    import cv2
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = 20; params.maxThreshold = 230; params.thresholdStep = 5
    params.minDistBetweenBlobs = 5
    params.filterByColor = False
    params.filterByArea = True; params.minArea = 60; params.maxArea = 300
    params.filterByCircularity = True; params.minCircularity = 0.75
    params.filterByConvexity = True; params.minConvexity = 0.7
    params.filterByInertia = True; params.minInertiaRatio = 0.3
    return cv2.SimpleBlobDetector_create(params)


def detect_sbd(module_img, sbd, center_tol=0.35):
    module = module_img.astype(np.uint8)
    height, width = module.shape
    if height < 8: return 0

    corner_pixels = np.concatenate([
        module[:height//6, :width//6].ravel(), module[:height//6, -width//6:].ravel(),
        module[-height//6:, :width//6].ravel(), module[-height//6:, -width//6:].ravel(),
    ])
    bg_dark = corner_pixels.mean() < 128
    img = 255 - module if bg_dark else module

    keypoints = sbd.detect(img)
    for kp in keypoints:
        if abs(kp.pt[0] - width/2) <= center_tol * width and abs(kp.pt[1] - height/2) <= center_tol * height:
            return 1
    return 0


def classify_qr(image_path, source="denoised", return_extras=False, mode="ensemble"):
    gray = load_gray(image_path)
    pre = preprocess(gray)
    src = pre[source]
    ms, oy, ox = pre["ms"], pre["oy"], pre["ox"]

    blob = np.zeros((QR_SIZE, QR_SIZE), dtype=np.uint8)
    scores = np.zeros((QR_SIZE, QR_SIZE), dtype=np.float32)
    sbd = make_sbd() if mode in ("sbd", "ensemble") else None

    for r in range(QR_SIZE):
        for c in range(QR_SIZE):
            module = extract_module(src, ms, oy, ox, r, c)
            if module.size == 0: continue
            if mode == "log_shape":
                detection = detect_blob_square(module)
                blob[r, c] = int(detection["has_blob"])
                scores[r, c] = detection["score"]
            elif mode == "sbd":
                blob[r, c] = detect_sbd(module, sbd)
            else:  
                sbd_yes = detect_sbd(module, sbd)
                detection = detect_blob_square(module)
                scores[r, c] = detection["score"]
                
                blob[r, c] = int(sbd_yes and detection["score"] > -3.0)

    if return_extras:
        return blob, scores, pre
    return blob


try:
    from reedsolo import RSCodec, ReedSolomonError
    rs = RSCodec(19)     
    SIG_BITS = 664       

    def bits_to_bytes(bits):
        n_full = len(bits) - len(bits) % 8
        out = bytearray()
        for i in range(0, n_full, 8):
            byte = 0
            for bit in bits[i:i + 8]:
                byte = (byte << 1) | int(bit)
            out.append(byte)
        return bytes(out)

    def try_decode(blob_matrix):
        bits = [int(blob_matrix[r, c])
                for r in range(QR_SIZE)
                for c in range(QR_SIZE)
                if not FUNCTIONAL[r, c]]
        if len(bits) < SIG_BITS:
            return None
        raw = bits_to_bytes(bits[:SIG_BITS])
        try:
            decoded = rs.decode(raw)
            sig = bytes(decoded[0] if isinstance(decoded, tuple) else decoded)
            return sig if len(sig) == 64 else None
        except ReedSolomonError:
            return None
except ImportError:
    def try_decode(_):
        return None
