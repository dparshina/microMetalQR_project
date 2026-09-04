import warnings

import cv2

from localize_and_warp import primary_pipeline, fallback_pipeline

warnings.filterwarnings("ignore")

from paths import RAW_DIR, CACHE_DIR

OUT_DIR = CACHE_DIR / 'warped'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def warp_cascade(image_path):
    warped, method, ms = primary_pipeline(image_path)
    if warped is not None:
        return warped, f'primary_{method}', ms

    warped, method, ms = fallback_pipeline(image_path)
    if warped is not None:
        return warped, f'fallback_{method}', ms

    return None, 'fail', ms


def main():
    images = []
    for folder_dir in sorted(RAW_DIR.iterdir()):
        if not folder_dir.is_dir(): continue
        for path in sorted(folder_dir.glob('*.jpg')):
            images.append((folder_dir.name, path))
    print(f'Found {len(images)} raw images', flush=True)

    n_saved = 0
    for i, (folder_name, path) in enumerate(images):
        try:
            warped, _, _ = warp_cascade(path)
        except Exception:
            warped = None

        if warped is not None:
            out_path = OUT_DIR / folder_name / f'{path.stem}.png'
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path), warped)
            n_saved += 1

        if (i + 1) % 20 == 0:
            print(f'  [{i+1:3d}/{len(images)}]', flush=True)

    print(f'\nWrote {n_saved} rectified crops to: {OUT_DIR}')


main()
