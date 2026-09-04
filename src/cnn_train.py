import json, time, argparse
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from cnn_models import VARIANTS, n_params
from decode_pipeline import load_gray, preprocess, FUNCTIONAL, try_decode, QR_SIZE
from cnn_build_dataset import extract_module_resized

from paths import ROOT, WARP_DIR, GT_FILE, CACHE_DIR, WEIGHTS_DIR, RES_FILE

CACHE_DIR = CACHE_DIR / 'attn_cache'
CACHE_DIR.mkdir(parents=True, exist_ok=True)
WEIGHTS_DIR.mkdir(exist_ok=True)
RES_FILE.parent.mkdir(exist_ok=True)

EXPECTED = 'b2e2bb5898e429e5d5adfd8d34186738bb6699a110304807f03c9edd1ff81bc1e904094a718b9f7be2c80f0bbfa51554c54875546bd4cb852fa9b41b8aacf96d'

MODULE_SIZE_PX = 48
WINDOW_SCALE = 1.2
BATCH = 256
EPOCHS = 20
LR = 1.5e-3
WD = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
VAL_FRAC = 0.3
NUM_WORKERS = 2

PATTERNS = ['triangle', 'corner', 'square', 'cross']
SIZE = 's15'
N_FOLDS = 5                 
CV_SHUFFLE_SEED = 42        

GT = np.zeros((QR_SIZE, QR_SIZE), dtype=np.uint8)
with open(GT_FILE) as f:
    for line in f:
        if line.strip():
            r, c = [int(x) for x in line.strip().split(',')]
            GT[r, c] = 1

USED = np.zeros((QR_SIZE, QR_SIZE), dtype=bool)
data_cell_count = 0
for r in range(QR_SIZE):
    for c in range(QR_SIZE):
        if not FUNCTIONAL[r, c]:
            if data_cell_count < 664: USED[r, c] = True
            data_cell_count += 1

EXCLUDE = {'PHOTO_1777551991 copy.png', 'PHOTO_1777551995 copy.png'}


def list_images(pattern):
    folder = WARP_DIR / f'{pattern}_{SIZE}'
    if not folder.exists(): return []
    return [p for p in sorted(folder.glob('*.png'))
            if 'qr_final_' not in p.name and p.name not in EXCLUDE]


def build_dataset(pattern):
    cache_x = CACHE_DIR / f'{pattern}_X.npy'; cache_y = CACHE_DIR / f'{pattern}_y.npy'
    cache_meta = CACHE_DIR / f'{pattern}_meta.json'
    if cache_x.exists() and cache_y.exists() and cache_meta.exists():
        return np.load(cache_x), np.load(cache_y), json.loads(cache_meta.read_text())

    images = list_images(pattern)
    patches, labels, records = [], [], []
    for image_idx, path in enumerate(images):
        pre = preprocess(load_gray(path))
        for r in range(QR_SIZE):
            for c in range(QR_SIZE):
                patch = extract_module_resized(pre['denoised'], pre['ms'], pre['oy'], pre['ox'],
                                               r, c, WINDOW_SCALE, MODULE_SIZE_PX)
                patches.append(patch); labels.append(int(GT[r, c]))
                records.append({'image_idx': image_idx, 'r': r, 'c': c})

    X = np.stack(patches, 0).astype(np.uint8); y = np.array(labels, np.uint8)
    np.save(cache_x, X); np.save(cache_y, y)
    meta = {'records': records, 'image_names': [str(path.relative_to(ROOT)) for path in images]}
    cache_meta.write_text(json.dumps(meta))
    return X, y, meta


class ModuleDataset(Dataset):
    def __init__(self, X, y, idx, augment, pattern):
        self.X, self.y, self.idx, self.aug = X, y, idx, augment
        self.full_rot = pattern in ('corner', 'triangle', 'cross')

    def __len__(self): 
        return len(self.idx)

    def __getitem__(self, i):
        k = self.idx[i]
        img = self.X[k].astype(np.float32) / 255.0
        if self.aug: img = self.augment_image(img)
        img = (img - img.mean()) / (img.std() + 1e-6)
        return torch.from_numpy(img).unsqueeze(0), torch.tensor(float(self.y[k]))

    def augment_image(self, img):
        H, W = img.shape
        dy = np.random.randint(-int(H*0.1), int(H*0.1)+1)
        dx = np.random.randint(-int(W*0.1), int(W*0.1)+1)
        img = np.roll(img, (dy, dx), (0,1))

        img = np.clip(img * np.random.uniform(0.75,1.25) + np.random.uniform(-0.15,0.15), 0, 1)

        if np.random.rand() < 0.5:
            img = np.clip(img + np.random.randn(H,W).astype(np.float32)*0.03, 0, 1)

        if np.random.rand() < 0.4:
            for _ in range(np.random.randint(1,3)):
                sz = np.random.randint(4,9); yy = np.random.randint(0,H-sz); xx = np.random.randint(0,W-sz)
                img[yy:yy+sz, xx:xx+sz] = np.random.uniform(0,1)

        if self.full_rot:
            k = np.random.randint(0,4)
            if k: img = np.rot90(img, k).copy()
            if np.random.rand() < 0.5: img = np.fliplr(img).copy()
            if np.random.rand() < 0.5: img = np.flipud(img).copy()
        else:
            if np.random.rand() < 0.5: img = np.fliplr(img).copy()
            if np.random.rand() < 0.5: img = np.flipud(img).copy()

        return img


def predict_blob(model, X, recs, image_idx):
    blob = np.zeros((QR_SIZE, QR_SIZE), dtype=np.uint8)
    record_indices = [i for i, record in enumerate(recs) if record['image_idx'] == image_idx]

    patches, coords = [], []
    for k in record_indices:
        patch = X[k].astype(np.float32) / 255.0
        patch = (patch - patch.mean()) / (patch.std() + 1e-6)
        patches.append(patch); coords.append((recs[k]['r'], recs[k]['c']))
    batch = torch.from_numpy(np.stack(patches)).unsqueeze(1).to(DEVICE)

    model.eval(); batch_outputs = []
    with torch.no_grad():
        for start in range(0, len(batch), BATCH):
            batch_outputs.append(model(batch[start:start+BATCH]).cpu().numpy())
    logits = np.concatenate(batch_outputs)

    for (r, c), logit in zip(coords, logits):
        if USED[r, c]: blob[r, c] = int(logit > 0)
    return blob


def count_byte_errors(blob):
    bits_pred = [int(blob[r,c]) for r in range(QR_SIZE) for c in range(QR_SIZE) if not FUNCTIONAL[r,c]][:664]
    bits_gt = [int(GT[r,c]) for r in range(QR_SIZE) for c in range(QR_SIZE) if not FUNCTIONAL[r,c]][:664]
    return sum(1 for i in range(0, 664, 8) if bits_pred[i:i+8] != bits_gt[i:i+8])


def evaluate_from_checkpoint(pattern, variant, fold, X, y, recs, names, weights_path):
    _, val_images = cv_split(len(names))[fold]
    model = VARIANTS[variant](in_size=MODULE_SIZE_PX).to(DEVICE)
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE)); model.eval()

    per_image = []; decoded = 0
    for img_idx in val_images:
        blob = predict_blob(model, X, recs, img_idx)
        byte_errors = count_byte_errors(blob); sig = try_decode(blob)
        ok = bool(sig and sig.hex() == EXPECTED)
        if ok: decoded += 1
        per_image.append({'image': names[img_idx], 'byte_err': byte_errors, 'decoded': ok, 'blob_count': int(blob.sum())})
    return {
        'pattern': pattern, 'variant': variant, 'fold': fold,
        'n_val': len(val_images), 'decoded': decoded,
        'decode_rate': decoded / len(val_images),
        'best_f1': None, 'curves': None,
        'per_image': per_image,
        'val_image_names': [names[i] for i in val_images],
        'n_params': n_params(model),
    }


def cv_split(n_images, n_folds=N_FOLDS, shuffle_seed=CV_SHUFFLE_SEED):
    rng = np.random.RandomState(shuffle_seed)
    order = np.arange(n_images); rng.shuffle(order)
    folds = np.array_split(order, n_folds)
    splits = []
    for k in range(n_folds):
        val = list(folds[k])
        train = [i for j, f in enumerate(folds) if j != k for i in f]
        splits.append((train, val))
    return splits


def run_single_fold(pattern, variant, fold, X, y, recs, names, save_weights=True):
    n_images = len(names)
    patches_by_image = [[] for _ in range(n_images)]
    for i, r in enumerate(recs): patches_by_image[r['image_idx']].append(i)
    splits = cv_split(n_images)
    train_images, val_images = splits[fold]
    train_idx = [i for j in train_images for i in patches_by_image[j]]
    val_idx = [i for j in val_images for i in patches_by_image[j]]


    torch.manual_seed(fold); np.random.seed(fold)
    train_ds = ModuleDataset(X, y, train_idx, True, pattern)
    val_ds = ModuleDataset(X, y, val_idx, False, pattern)
    train_dl = DataLoader(train_ds, BATCH, shuffle=True, num_workers=NUM_WORKERS, persistent_workers=NUM_WORKERS>0)
    val_dl = DataLoader(val_ds, BATCH, shuffle=False, num_workers=NUM_WORKERS, persistent_workers=NUM_WORKERS>0)

    model = VARIANTS[variant](in_size=MODULE_SIZE_PX).to(DEVICE)

    pos_rate = y[train_idx].mean()
    pos_weight = torch.tensor([(1-pos_rate)/max(pos_rate,1e-6)]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    curves = {'train_loss': [], 'val_loss': [], 'val_f1': [], 'val_decode': []}
    best_f1 = -1; best_state = None
    for epoch in range(EPOCHS):
        model.train(); train_loss_sum = 0
        for xb, yb in train_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            loss = criterion(model(xb), yb)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            train_loss_sum += loss.item()
        scheduler.step()

        model.eval(); preds, labels, val_loss_sum = [], [], 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb = xb.to(DEVICE); yb_d = yb.to(DEVICE)
                logits = model(xb)
                val_loss_sum += criterion(logits, yb_d).item()
                preds.extend((logits.cpu().numpy() > 0).astype(int))
                labels.extend(yb.numpy().astype(int))
        preds, labels = np.array(preds), np.array(labels)
        tp = int(((preds==1)&(labels==1)).sum())
        fp = int(((preds==1)&(labels==0)).sum())
        fn = int(((preds==0)&(labels==1)).sum())
        prec = tp/max(tp+fp,1); rec = tp/max(tp+fn,1)
        f1 = 2*prec*rec/max(prec+rec, 1e-9)

        decoded_count = 0
        for img_idx in val_images:
            blob = predict_blob(model, X, recs, img_idx)
            sig = try_decode(blob)
            if sig and sig.hex() == EXPECTED: decoded_count += 1
        curves['train_loss'].append(train_loss_sum/len(train_dl)); curves['val_loss'].append(val_loss_sum/len(val_dl))
        curves['val_f1'].append(f1); curves['val_decode'].append(decoded_count / len(val_images))

        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.clone().cpu() for k,v in model.state_dict().items()}

    model.load_state_dict(best_state)

    per_image = []
    decoded = 0
    for img_idx in val_images:
        blob = predict_blob(model, X, recs, img_idx)
        byte_errors = count_byte_errors(blob)
        sig = try_decode(blob)
        ok = bool(sig and sig.hex() == EXPECTED)
        if ok: decoded += 1
        per_image.append({'image': names[img_idx], 'byte_err': byte_errors, 'decoded': ok, 'blob_count': int(blob.sum())})
    if save_weights:
        torch.save(best_state, WEIGHTS_DIR / f'{variant}_{pattern}_f{fold}.pt')
    return {
        'pattern': pattern, 'variant': variant, 'fold': fold,
        'n_val': len(val_images), 'decoded': decoded,
        'decode_rate': decoded / len(val_images),
        'best_f1': float(best_f1),
        'curves': curves,
        'per_image': per_image,
        'val_image_names': [names[i] for i in val_images],
        'n_params': n_params(model),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variants', nargs='+', default=list(VARIANTS.keys()))
    parser.add_argument('--patterns', nargs='+', default=PATTERNS)
    parser.add_argument('--folds', nargs='+', type=int, default=list(range(N_FOLDS)))
    args = parser.parse_args()

    print(f'device={DEVICE}  | 5-fold CV (shuffle seed={CV_SHUFFLE_SEED})', flush=True)
    print('=== building datasets ===', flush=True)
    datasets = {}
    for pattern in args.patterns:
        start_time = time.time()
        X, y, meta = build_dataset(pattern)
        datasets[pattern] = (X, y, meta['records'], meta['image_names'])
        print(f'  {pattern}: N_img={len(meta["image_names"])} N_mod={len(X)} pos_rate={y.mean():.3f}  ({time.time()-start_time:.0f}s)', flush=True)

    runs = []
    if RES_FILE.exists():
        runs = json.loads(RES_FILE.read_text()).get('runs', [])
    done = {(run['pattern'], run['variant'], run['fold']) for run in runs}

    total = len(args.variants) * len(args.patterns) * len(args.folds)
    run_num = 0
    for variant in args.variants:
        for pattern in args.patterns:
            for fold in args.folds:
                run_num += 1
                key = (pattern, variant, fold)
                if key in done:
                    print(f'[{run_num}/{total}] {variant}/{pattern}/fold{fold} cached (json)', flush=True); continue

                
                weights_path = WEIGHTS_DIR / f'{variant}_{pattern}_f{fold}.pt'
                if weights_path.exists():
                    print(f'[{run_num}/{total}] {variant}/{pattern}/fold{fold} eval-only from .pt', flush=True)
                    start_time = time.time()
                    X, y, recs, names = datasets[pattern]
                    result = evaluate_from_checkpoint(pattern, variant, fold, X, y, recs, names, weights_path)
                    result['time_s'] = round(time.time()-start_time, 1); result['resumed'] = True
                    runs.append(result)
                    RES_FILE.write_text(json.dumps({'runs': runs}, indent=1, default=str))
                    print(f'  → decoded {result["decoded"]}/{result["n_val"]} = {result["decode_rate"]:.1%}  ({result["time_s"]:.0f}s)', flush=True)
                    continue

                print(f'\n[{run_num}/{total}] {variant}/{pattern}/fold{fold}', flush=True)
                start_time = time.time()
                X, y, recs, names = datasets[pattern]
                result = run_single_fold(pattern, variant, fold, X, y, recs, names)
                result['time_s'] = round(time.time()-start_time, 1)
                runs.append(result)
                RES_FILE.write_text(json.dumps({'runs': runs}, indent=1, default=str))
                print(f'  → decoded {result["decoded"]}/{result["n_val"]} = {result["decode_rate"]:.1%}  f1={result["best_f1"]:.3f}  ({result["time_s"]:.0f}s)', flush=True)

    print('\n\n=== SUMMARY (5-fold CV) ===')
    print(f'{"variant":<10} {"pattern":<10} {"folds":<6} {"decode_rate":<18} {"params":<8}')
    rates_by_key = defaultdict(list)
    for run in runs: rates_by_key[(run['variant'], run['pattern'])].append(run['decode_rate'])
    for variant in args.variants:
        for pattern in args.patterns:
            rates = rates_by_key.get((variant, pattern), [])
            if not rates: continue
            param_count = next((run['n_params'] for run in runs if run['variant']==variant), '?')
            print(f'{variant:<10} {pattern:<10} {len(rates):<6} {np.mean(rates):>5.1%} ± {np.std(rates):>4.1%}     {param_count:>6,}')


main()
