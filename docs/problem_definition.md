# Problem definition

## Title

Fresh and Rotten Fruit Classification with Convolutional Neural Networks

## Motivation

Checking fruit quality before distribution is still largely manual. An image-based classifier that
recognises the fruit **and** tells whether it is still fresh can support automatic sorting and
quality control (the purpose stated by the dataset authors).

## Research objectives

1. Build and compare three CNN models of increasing sophistication on the same data:
   * **Model 1** - a simple sequential CNN (Conv-ReLU-MaxPool blocks + fully connected layers)
     that predicts the 16 combined classes (fruit x freshness);
   * **Model 2** - a multi-task CNN (Keras Functional API) with a shared residual trunk and two
     heads: fruit type (8-way softmax) and freshness (sigmoid);
   * **Model 3** - transfer learning: an ImageNet-pretrained MobileNetV2 with the same two heads,
     trained with a frozen backbone and then fine-tuned from its last blocks.
2. Measure the effect of data augmentation and of the number of fine-tuned layers (ablations).
3. Use Grad-CAM to check which image regions drive the fresh/spoiled decision.

## Input

One RGB photograph of a single fruit, 224 x 224 pixels.

## Output

* **Fruit type**: one of 8 classes - Banana, Lemon, Lulo, Mango, Orange, Strawberry, Tamarillo, Tomato.
* **Freshness**: fresh or spoiled, with P(spoiled). **Spoiled is the positive class** (the
  quality-control goal is to catch spoiled fruit, so recall on spoiled matters most).
* Equivalently a **combined label** among 16 classes (e.g. `Mango_spoiled`);
  `combined = fruit_label * 2 + freshness_label`.

## Evaluation

All models are evaluated on the same held-out test split with three label views:
fruit (8 classes), freshness (binary, positive = spoiled) and joint (16 classes, correct only if
both are right). Metrics: accuracy, precision, recall, F1 (macro for multi-class), ROC-AUC for
freshness, confusion matrices, and inference time per image. Model 1's 16-class output is decoded
into fruit and freshness so the three models are compared on identical terms.

## Summary of tasks

1. Data audit: readability, image statistics, exact duplicates (SHA-256), label conflicts.
2. Cleaning and a leakage-aware stratified split (70/15/15) that keeps near-identical burst
   frames in the same subset.
3. Shared input pipeline: decoding, normalisation (inside each model), augmentation for training only.
4. Design, train and tune the three models with the same callbacks (early stopping, best checkpoint,
   learning-rate reduction).
5. Ablations: with/without augmentation; fine-tuning depth of MobileNetV2.
6. Test-set evaluation, comparison table, confusion matrices, inference time.
7. Grad-CAM visualisation and error analysis.

## Constraints

* Course tech stack: Python, TensorFlow/Keras, scikit-learn, NumPy, pandas, matplotlib.
* Training on a laptop CPU (Intel i9-13900H); TensorFlow on native Windows has no GPU support.
