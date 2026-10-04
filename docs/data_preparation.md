# Data preparation

Reproduce everything below with:

```bash
python scripts/download_data.py   # download + extract the dataset (100 MB)
python scripts/audit_data.py      # audit reports and figures (reads all 16,000 images, ~3 min)
python scripts/make_splits.py     # cleaning decisions, groups, split files, split audit
```

## 1. Dataset

* **Spoiled and fresh fruit inspection dataset** (FRUIT-16K) - Pachon Suescun, Pinzon Arenas,
  Jimenez-Moreno (Universidad Militar Nueva Granada), Mendeley Data, 2020.
  <https://data.mendeley.com/datasets/6ps7gtp2wg/1>, DOI 10.17632/6ps7gtp2wg.1, licence CC BY 4.0.
* 16,000 JPEG images, 16 folders `F_<Fruit>` (fresh) / `S_<Fruit>` (spoiled), 1,000 images each,
  files `1.jpg ... 1000.jpg` in capture order.
* 8 fruits: banana, lemon, lulo, mango, orange, strawberry, tamarillo, tomato.
* Every image is **224 x 224 RGB**, about 6 KB (heavily compressed), photographed on several
  backgrounds (steel sink, wooden table, notebook, tablecloth, paper) with varying distance,
  rotation and lighting.
* Mean / std of pixel values (after cleaning, [0, 1] scale): R 0.494 / 0.209, G 0.424 / 0.195, B 0.389 / 0.197.

## 2. Audit findings (reports/)

| Check | Result |
|---|---|
| Unreadable / corrupt files | 0 |
| Image size / colour mode | all 224 x 224, all RGB |
| Exact duplicates (same SHA-256) | **1,273 copies**, always consecutive pairs (e.g. 99.jpg = 100.jpg): F_Lemon 445, F_Lulo 442, F_Tamarillo 318, S_Tomato 68 |
| Label conflicts | **1**: `F_Banana/1.jpg` is the same photo as `S_Banana/6.jpg` (pHash distance 0); the banana is visibly spoiled |
| Near-duplicates | images were shot in bursts, so neighbouring files are often near-copies (see figures/duplicate_examples.png) |

## 3. Cleaning

* Removed the 1,273 duplicate copies (the first file of each identical pair is kept).
* Removed `F_Banana/1.jpg` (mislabelled).
* Result: **14,726 images**. Classes are now mildly imbalanced: Lemon_fresh 555, Lulo_fresh 558,
  Tamarillo_fresh 682, Tomato_spoiled 932, Banana_fresh 999, all others 1,000.

## 4. Why a plain random split is not enough

Because of the burst capture, a random split puts near-identical frames of the same fruit into
both train and test, so the test score would partly measure memorisation. We measured, for every
test image, how close the most similar training image of the same class is
(perceptual hash = pHash, 64-bit; distance 0 = identical structure):

| Split method | test images with a near-copy in train (pHash <= 6) | pHash <= 10 |
|---|---:|---:|
| Random stratified split (image level) | 9.7 % | 32.9 % |
| Blocks of 25 consecutive frames kept together | 3.9 % | 20.3 % |
| **Final: blocks of 25 + near-duplicate merging** | **1.2 %** | **14.6 %** |

## 5. Final split method (leakage-aware, stratified)

1. **Groups.** Inside each class folder, every 25 consecutive files form a group (one burst).
   Two images of the same class are also put in the same group when they are near-duplicates:
   pHash distance <= 6 **and** either nearly identical pixels (16 x 16 thumbnail mean absolute
   difference <= 0.03) or at most 50 files apart (the same fruit re-shot under different light).
2. **Split.** Whole groups are assigned to train / validation / test **separately for each of the
   16 classes** (stratification), largest groups first, each to the subset with the most room
   left (target 70 / 15 / 15). Seed 42, fully deterministic.
3. **Checks** (reports/split_audit.txt): no file, SHA-256 or group appears in two subsets; no
   near-duplicate pair crosses subsets; per-class shares within 1.3 points of the target.

| Subset | Images | Share | Used for |
|---|---:|---:|---|
| Training | 10,320 | 70.1 % | fitting the weights |
| Validation | 2,203 | 15.0 % | early stopping, checkpoint selection, learning-rate schedule, ablation decisions |
| Test | 2,203 | 15.0 % | final evaluation only (never used for any decision) |

Per-class counts: reports/class_distribution.csv, figures/class_distribution.png.

## 6. Pre-processing pipeline (src/data_pipeline.py)

* **Decoding / resizing**: JPEG -> RGB tensor 224 x 224 x 3 (the native size, so no resizing is needed).
* **Normalisation** is the first layer of each model, so saved models accept raw pixels:
  Models 1-2 `Rescaling(1/255)` -> [0, 1]; Model 3 MobileNetV2 `x / 127.5 - 1` -> [-1, 1]
  (the range its ImageNet weights were trained with).
* **Augmentation (training split only)**, Keras preprocessing layers, new random values every epoch:
  horizontal + vertical flip, rotation up to +-36 degrees (factor 0.1), zoom +-15 %, brightness
  +-15 %, contrast +-15 %. No hue/saturation changes, because colour (browning, dark spots, mould)
  is the main freshness cue.
* tf.data: cache the encoded JPEG bytes in memory (~60 MB for the training split), shuffle every
  epoch, decode, batch 32, prefetch.
