import json

import numpy as np
from PIL import Image

from decode_pipeline import load_gray, preprocess, FUNCTIONAL, QR_SIZE
from paths import ROOT, WARP_DIR, GT_FILE, CACHE_DIR

OUT_DIR = CACHE_DIR / 'cnn_dataset'
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODULE_SIZE_PX = 48     
WINDOW_SCALE = 1.2      

EXCLUDE = {'PHOTO_1777551991 copy.png', 'PHOTO_1777551995 copy.png'}


gt_blobs = np.zeros((QR_SIZE, QR_SIZE), dtype=np.uint8)
with open(GT_FILE) as f:
    for line in f:
        line = line.strip()
        if line:
            row, col = [int(x) for x in line.split(',')]
            gt_blobs[row, col] = 1


payload_mask = np.zeros((QR_SIZE, QR_SIZE), dtype=bool)
data_cell_count = 0
for row in range(QR_SIZE):
    for col in range(QR_SIZE):
        if not FUNCTIONAL[row, col]:
            if data_cell_count < 664:
                payload_mask[row, col] = True
            data_cell_count += 1

print(f'payload data modules: {payload_mask.sum()}  (functional: {FUNCTIONAL.sum()})')
print(f'ground-truth blobs: {gt_blobs.sum()}  (in payload: {((gt_blobs == 1) & payload_mask).sum()})')


def extract_module_resized(img, ms, oy, ox, r, c, scale, out_size):
    center_y = oy + (r + 0.5) * ms
    center_x = ox + (c + 0.5) * ms
    half = ms * scale / 2.0
    height, width = img.shape

    y0 = int(round(center_y - half)); y1 = int(round(center_y + half))
    x0 = int(round(center_x - half)); x1 = int(round(center_x + half))

    pad_top    = max(0, -y0); pad_bottom = max(0, y1 - height)
    pad_left   = max(0, -x0); pad_right  = max(0, x1 - width)
    y0_clip, y1_clip = max(0, y0), min(height, y1)
    x0_clip, x1_clip = max(0, x0), min(width, x1)

    crop = img[y0_clip:y1_clip, x0_clip:x1_clip]
    if pad_top or pad_bottom or pad_left or pad_right:
        crop = np.pad(crop, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='reflect')

    patch = Image.fromarray(np.clip(crop, 0, 255).astype(np.uint8))
    return np.array(patch.resize((out_size, out_size), Image.LANCZOS))


def list_images():
    images = []
    for folder in ['square_s10', 'square_s15']:
        for path in sorted((WARP_DIR / folder).glob('*.png')):
            if 'qr_final_' in path.name: continue
            if path.name in EXCLUDE: continue
            images.append(path)
    return images


def main():
    images = list_images()
    print(f'Processing {len(images)} images...')

    all_patches = []   
    all_labels = []   
    all_records = []  

    for image_idx, path in enumerate(images):
        gray = load_gray(path)
        pre = preprocess(gray)
        ms, oy, ox = pre['ms'], pre['oy'], pre['ox']
        
        denoised = pre['denoised']

        for row in range(QR_SIZE):
            for col in range(QR_SIZE):
                patch = extract_module_resized(denoised, ms, oy, ox, row, col,
                                               WINDOW_SCALE, MODULE_SIZE_PX)
                all_patches.append(patch)
                all_labels.append(int(gt_blobs[row, col]))
                all_records.append({
                    'image': path.name,
                    'image_idx': image_idx,
                    'r': row, 'c': col,
                    'used': bool(payload_mask[row, col]),
                    'functional': bool(FUNCTIONAL[row, col]),
                })
        print(f'  [{image_idx+1:2d}/{len(images)}] {path.parent.name}/{path.name:<35} ms={ms:.1f}')

    X = np.stack(all_patches, axis=0)
    y = np.array(all_labels, dtype=np.uint8)
    print(f'\nDataset: X={X.shape} y={y.shape} positive_rate={y.mean():.3f}')

    np.save(OUT_DIR / 'X.npy', X)
    np.save(OUT_DIR / 'y.npy', y)
    with open(OUT_DIR / 'meta.json', 'w') as f:
        json.dump({
            'module_size_px': MODULE_SIZE_PX,
            'window_scale': WINDOW_SCALE,
            'n_images': len(images),
            'images': [str(path.relative_to(ROOT)) for path in images],
            'records': all_records,
        }, f, indent=1)
    print(f'Saved to {OUT_DIR}')


if __name__ == '__main__':
    main()
