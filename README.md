## Code, data and results for the computer-vision part of

> **AuthenTag: Vision-Enabled IoT and Permissioned Blockchain Architecture for
> Authenticated Supply Chain Event Tracking**
> Y. Madhwal, D. Parshina, A. Sivolotskii, A. Barabulya, I. Nosov, A. Korotkevich,
> A. Chekanov, A. Abdurashitov, Y. Yanovich, G. Sukhorukov.
> Submitted to *IEEE Internet of Things Journal*, 2026.

The paper describes a dual-layer QR marker: a conventional QR payload plus a 
second layer that carries a Reed–Solomon protected ECDSA signature
embedded in the geometry of the QR data modules. This repository contains
everything needed to reproduce **Section VI (Experimental Evaluation)** — the
marker encoder, the localization/rectification pipeline, the module classifiers.

The blockchain side of the system (Go IoT gateway, Hyperledger Fabric chaincode,
Raspberry Pi node, web interface) is **not** part of this repository.

---

## Repository layout

```
data/
  raw/<geometry>_<scale>/*.jpg     406 photographs of laser-engraved markers
  raw/<geometry>_<scale>/qr_final_*.png   the encoder rendering of each marker
  warped/<geometry>_<scale>/*.png  389 rectified grayscale crops (stage-1 output)
  ground_truth_blobs.txt           337 hidden-bit module coordinates (r,c)
src/
  qrgen_encoder.ipynb              Algorithm 1 — dual-layer marker generation
  localize_and_warp.py             QR localization cascade + perspective warp
  localize_run_all.py              stage 1 — batch rectification of data/raw
  decode_pipeline.py               preprocessing, classical blob classifier, RS decoding
  cnn_models.py                    BlobCNN, BlobCNN-SE, BlobCNN-SpatialPre
  cnn_build_dataset.py             module-patch extraction (48×48, 1.2× window)
  cnn_train.py                     stage 2 — 5-fold CV training and evaluation
  paths.py                         every path used by the scripts
models/                            60 trained checkpoints, <variant>_<geometry>_f<fold>.pt
results/
  cnn_attention_results.json       per-run curves, per-image decode outcomes
  figures/                         summary bar, per-pattern curves,
                                   per-image decode rate, module overlays
```

`cache/` (created on demand) holds derived artifacts — extracted module patches
and re-run stage-1 output. It is regenerated from `data/` and is not tracked.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Reproducing the pipeline

**Stage 1: localization and rectification.** Runs the cascade (heavy-denoise
preprocessing + OpenCV QR detector, then lighter binarization, the WeChat
detector, and contour-based finder detection) over all 406 photographs:

```bash
python src/localize_run_all.py
```

It writes rectified images and a per-image CSV to `cache/warped/`, leaving the
reference copies in `data/warped/` untouched. The shipped run localizes 389 of
406 images, the 303 rectified `*_s15` images are the corpus used for the
classifier experiments.

**Stage 2: classifier training and 5 fold cross validation.**

```bash
python src/cnn_train.py                     
python src/cnn_train.py --patterns triangle --variants spatial --folds 0
```

Folds are image-level (a fixed permutation with seed 42 splits the images of one
geometry into five disjoint folds), so patches from one photograph never appear
in both training and validation. Runs already present in
`results/cnn_attention_results.json` are skipped, delete the file to retrain from
scratch. When a checkpoint exists in `models/` but the JSON entry is missing, the
script re-evaluates from the checkpoint instead of retraining.

The reported metric is the complete signature decode rate: predicted hidden bits
are packed into bytes, passed through the Reed–Solomon decoder (19 parity
symbols, up to 9 correctable bytes), and the run counts as successful only when
the recovered 64-byte ECDSA signature matches the reference exactly.

**Figures.** The figures reported in the paper ship under `results/figures/`
(summary bar, per-pattern curves, per-image decode rate, and module overlays),
generated from `results/cnn_attention_results.json`.

**Marker generation.** `src/qrgen_encoder.ipynb` builds the dual-layer symbols:
Version-4 QR (33×33, error correction level H), SHA-256 of the payload, ECDSA
signature over secp256k1, Reed–Solomon expansion, and rendering of the four
internal geometries. The signing key is generated on first run into
`private_key.pem` / `public_key.pem`, those files are excluded from the
repository.

## The dataset

Photographs of laser-engraved metallic markers taken with an iPhone 14 Pro and a
macro lens under diffuse illumination and varying capture conditions. Each folder
is `<geometry>_<scale>`, where geometry is `triangle`, `corner`, `square` or
`cross`, and the scale is the encoder module rendering size (`s10` or `s15`
pixels per QR module). All photographs show the same physical payload, so a
single ground-truth map of 337 hidden-bit modules applies to every image.

See `data/README.md` for file naming and the ground-truth format.

## Known limitations

As reported in the paper, the dominant failure mode is geometric: a one- or
two-module misalignment of the reconstructed grid corrupts whole rows or columns
and exceeds the 9 byte Reed–Solomon correction budget, even when local module
classification is otherwise accurate.

## License

Code: MIT (`LICENSE`). Dataset (`data/`): CC BY 4.0 — please cite the paper if
you use the images.
