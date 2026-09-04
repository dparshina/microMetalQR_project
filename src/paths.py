from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'
RAW_DIR = DATA_DIR / 'raw'
WARP_DIR = DATA_DIR / 'warped'
GT_FILE = DATA_DIR / 'ground_truth_blobs.txt'
CACHE_DIR = ROOT / 'cache'
WEIGHTS_DIR = ROOT / 'models'
RESULTS_DIR = ROOT / 'results'
RES_FILE = RESULTS_DIR / 'cnn_attention_results.json'
FIG_DIR = RESULTS_DIR / 'figures'


def resolve_image(rel):
    p = Path(rel)
    direct = ROOT / p
    if direct.exists():
        return direct
    if len(p.parts) >= 2:
        return WARP_DIR / p.parts[-2] / p.name
    return direct
