# Model plan

All three models receive the same 224 x 224 x 3 RGB images from the same split, use the same
batch size (32), optimiser (Adam), callbacks and evaluation code. Code: `src/models/`,
training settings: `configs/training_config.json`, training entry point: `scripts/train.py`.

## Model 1 - simple sequential CNN (baseline, 16 classes)

Keras **Sequential** API, stacked Conv2D - ReLU - MaxPool blocks followed by fully connected layers
(the LeNet/AlexNet pattern from Chapter 5.1).

| Layer | Output shape | Params |
|---|---|---:|
| Input + Rescaling(1/255) | 224 x 224 x 3 | 0 |
| Conv 3x3, 32, ReLU + MaxPool 2x2 | 112 x 112 x 32 | 896 |
| Conv 3x3, 64, ReLU + MaxPool 2x2 | 56 x 56 x 64 | 18,496 |
| Conv 3x3, 128, ReLU + MaxPool 2x2 | 28 x 28 x 128 | 73,856 |
| Conv 3x3, 128, ReLU + MaxPool 2x2 | 14 x 14 x 128 | 147,584 |
| Conv 3x3, 256, ReLU + MaxPool 2x2 | 7 x 7 x 256 | 295,168 |
| Flatten | 12,544 | 0 |
| Dense 256, ReLU + Dropout 0.5 | 256 | 3,211,520 |
| Dense 16, softmax | 16 | 4,112 |
| **Total** | | **3,751,632** |

* Output: one softmax over the 16 combined classes; loss = sparse categorical cross-entropy.
* Fruit and freshness are decoded from the 16 probabilities: P(fruit) = P(fruit_fresh) + P(fruit_spoiled),
  P(spoiled) = sum of the 8 spoiled probabilities.
* Training: Adam, learning rate 1e-3, up to 40 epochs.

## Model 2 - multi-branch, multi-task CNN (Functional API)

A shared convolutional trunk built from **residual blocks** (ResNet-style CNN blocks with batch
normalisation, Chapter 5.2) splits into **two parallel heads** (multi-task learning).

```bash
image 224x224x3 -> Rescaling(1/255)
stem    Conv 3x3 stride 2 (32) - BN - ReLU - MaxPool 2x2      56 x 56 x 32
stage 1 ResBlock(64)                                         56 x 56 x 64
stage 2 ResBlock(128, stride 2)                              28 x 28 x 128
stage 3 ResBlock(256, stride 2)                              14 x 14 x 256
stage 4 ResBlock(512, stride 2)                               7 x 7 x 512
GlobalAveragePooling2D                                       512
  |-> Dense 128 ReLU - Dropout 0.3 - Dense 8 softmax   -> fruit
  '-> Dense 64  ReLU - Dropout 0.3 - Dense 1 sigmoid   -> freshness = P(spoiled)

ResBlock(f, s): Conv 3x3/s - BN - ReLU - Conv 3x3 - BN  (+)  shortcut -> ReLU
                shortcut = identity, or Conv 1x1/s + BN when the shape changes
```

* About 4.99 M parameters (one residual block per stage, i.e. a ResNet-10-style trunk).
* Loss = cross-entropy(fruit) + binary cross-entropy(freshness), equal weights (1.0 / 1.0).
* Why multi-task: fruit identity and freshness share low-level features (shape, texture, colour),
  but freshness is a binary decision that should not need 16 separate classes.
* Training: Adam, learning rate 1e-3, up to 40 epochs.

## Model 3 - transfer learning with MobileNetV2 (same two heads)

* Backbone: **MobileNetV2** pre-trained on ImageNet, `include_top=False` (the base network,
  Chapter 4.3 / 5.3), input preprocessing `x / 127.5 - 1` (the range its weights expect).
* Heads: exactly the two heads of Model 2 (same code, `src/models/heads.py`).
* **Stage 1 - feature extraction**: backbone frozen (2.26 M non-trainable parameters), only the
  heads train (247 k parameters); Adam 1e-3, up to 12 epochs.
* **Stage 2 - fine-tuning**: early layers stay frozen; the backbone is unfrozen from `block_13_expand`
  onwards (blocks 13-16 + the final 1x1 conv, 38 layers, 1.91 M trainable parameters);
  Adam 1e-4, up to 25 epochs. Batch-normalisation layers stay frozen (inference mode).
* Total 2.51 M parameters - the smallest model, but with ImageNet knowledge.

## Training protocol (identical for all models)

* Batch 32, augmentation on the training split only.
* Callbacks: ModelCheckpoint (best `val_loss`), EarlyStopping (patience 8, restore best weights),
  ReduceLROnPlateau (factor 0.3, patience 3, min 1e-6), CSVLogger, epoch timer.
* Seed 42. The test split is used once, at the end (`scripts/evaluate.py`).

## Ablations

| Ablation | Runs |
|---|---|
| Data augmentation | every model trained again without augmentation (`*_noaug`) |
| Fine-tuning depth (Model 3) | stage 2 restarted from the **same** stage-1 checkpoint, backbone unfrozen from: none (frozen), block 16, block 13 (main), block 10, block 6, all layers |

Run all experiments: `python scripts/run_experiments.py` - one training at a time, the three main
models first and then the ablations; runs that have already finished are skipped and an interrupted
run continues from its best checkpoint (`scripts/train.py --resume`).

## Evaluation and interpretation

* Test-set metrics for fruit, freshness (positive = spoiled) and joint 16-class prediction;
  confusion matrices; per-class precision/recall/F1; inference time per image on the CPU.
* Grad-CAM on the freshness output (last convolutional layer) to visualise which fruit regions
  drive the fresh/spoiled decision, including misclassified examples.
