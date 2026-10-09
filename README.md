# Computer Vision and Spatial AI

Five computer vision projects in PyTorch and related tools, covering custom architectures, pre-trained models, stress tests and real-time trackers. The training notebooks run on Apple Silicon (MPS) or CPU; the OpenFace and hand-tracking notebooks are written for Google Colab.

| Notebook | Task | Data | Main tools |
|---|---|---|---|
| [`image_captioning.ipynb`](image_captioning.ipynb) | Image captioning with visual attention | Flickr8k | PyTorch, spaCy, Hugging Face (BLIP) |
| [`object_detection.ipynb`](object_detection.ipynb) | Multi-class object detection and stress testing | Synthetic composites | PyTorch, Ultralytics YOLO11 |
| [`image_colorization.ipynb`](image_colorization.ipynb) | Grayscale-to-colour prediction | Oxford Flowers102 | PyTorch |
| [`hand_landmark_tracking.ipynb`](hand_landmark_tracking.ipynb) | Real-time hand tracking and gesture display | Webcam | MediaPipe |
| [`openface.ipynb`](openface.ipynb) | Face tracking robustness on in-the-wild video | UCF-101 | OpenFace 2.0 |

## 1. Image captioning (Flickr8k)

- **Split by image**, not by caption (80/10/10), so no image appears in two splits. The vocabulary is built from training captions only (spaCy tokenizer, minimum frequency 5).
- **Model:** a ResNet-50 encoder (ImageNet weights, fine-tuned at a lower learning rate) produces a 14×14 feature grid. An LSTM decoder with soft attention over that grid generates the caption (*Show, Attend and Tell* style), decoded with beam search (width 3).
- **Baseline:** the pre-trained [BLIP](https://huggingface.co/Salesforce/blip-image-captioning-base) model from Hugging Face, used zero-shot on the same test images.
- **Evaluation:** corpus BLEU-1 to BLEU-4 against all five reference captions per test image, plus attention-map visualisations.

| Model | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 |
|---|---|---|---|---|
| ResNet-50 + attention LSTM (trained on Flickr8k) | **69.2** | **50.5** | **37.1** | 26.5 |
| BLIP base (pre-trained, zero-shot) | 67.9 | 49.8 | 37.0 | **27.0** |

The custom model matches Flickr8k's annotation style, which BLEU rewards at the word level. BLIP is not fine-tuned on Flickr8k and uses a broader vocabulary that BLEU penalises, yet its Transformer decoder still wins on BLEU-4, which measures longer phrases. A Transformer decoder and a Vision Transformer encoder are possible next steps.

## 2. Object detection and stress testing (synthetic data)

### Setup

- **Dataset generator:** pastes one of three character sprites (`waldo`, `wenda`, `woof`) onto cartoon backgrounds collected with `icrawler`. It produces 5,000 train, 1,000 validation and 200 test images with YOLO-format labels.
- **Custom detector:** a ResNet-18 backbone with a classification head and a box-regression head, for one object per image.
- **YOLO11n:** fine-tuned from the Ultralytics checkpoint for 10 epochs.

| Model | Evaluated on | mAP@0.5 | mAP averaged over IoU thresholds |
|---|---|---|---|
| Custom ResNet-18 detector | Synthetic test (200 images) | 0.652 | 0.373* |
| YOLO11n | Synthetic test (200 images) | 0.995 | 0.991 |

\*IoU 0.5–0.9 in steps of 0.1. YOLO uses the standard 0.5–0.95 in steps of 0.05.

### Stress test (section 11)

The synthetic data above is easy by construction. The sprites are only rescaled (10–50% of the image), each image holds exactly one object, and all splits draw backgrounds from one shared pool. A near-perfect score therefore shows that YOLO learned the generator.

Section 11 builds a harder dataset (5,000 / 1,000 / 500 images) with:
- disjoint background pools per split;
- small objects (3–25% of the image side) and 0–3 objects per image, including empty images;
- flips, rotation, colour jitter, blur and JPEG artefacts;
- partial occlusion, with boxes shrunk to the visible part;
- unlabelled, hue-shifted look-alikes.

It then evaluates and retrains:

| Trained on | Evaluated on | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| Original synthetic | Original synthetic test | 0.998 | 1.000 | 0.995 | 0.991 |
| Original synthetic | Hard synthetic test | 0.924 | **0.530** | 0.577 | **0.353** |
| Hard synthetic | Hard synthetic test | 0.971 | 0.853 | 0.920 | 0.738 |
| Hard synthetic | Original synthetic test | 0.924 | 0.888 | 0.942 | 0.841 |

- **The 0.99 does not transfer.** On harder images, the model trained on easy data misses almost half of the objects (recall 0.53) and mAP50-95 falls from 0.99 to 0.35. Its precision stays at 0.92, so the main failure is missing small and occluded objects rather than false alarms.
- **Training on the harder data closes most of the gap** (0.74 mAP50-95 on the hard test) and generalises back to the easy data at 0.84.
- **The remaining drop on easy images most likely comes from scale.** The easy data contains larger objects (up to 50% of the image) than the hard training data covers (up to 25%).
- **No real-image evaluation yet.** All results are on synthetic data. Some backgrounds were crawled with queries such as *"waldo background"* and may contain the real characters without a label.

## 3. Image colorization (Flowers102)

Predicts RGB images from grayscale input at 64×64 resolution. Flowers102's larger official test and validation splits (7,169 images) are used for training and validation, and the 1,020-image official train split serves as the held-out test set. The notebook compares a convolutional network with residual blocks against a fully-connected network. Both are trained with L1 loss and early stopping, with flips, affine and perspective transformations as augmentation. The notebook visualises predictions on test images, CNN feature maps and the first-layer weights of the fully-connected network.

## 4. Real-time hand tracking (MediaPipe)

A Colab notebook that streams the webcam through MediaPipe's `HandLandmarker` task. It draws the 21 hand landmarks and classifies simple gestures (open hand, closed fist, thumbs up) from landmark geometry. An HTML dashboard overlays the gesture label and shows live metrics such as frames per second.

## 5. Face tracking robustness on in-the-wild video (OpenFace)

Runs OpenFace 2.0's `FaceLandmarkVidMulti` (CE-CLM landmark detector) on the first clip (`_c01`) of every group in all 101 UCF-101 actions: 2,525 videos. OpenFace writes per-frame landmarks, head pose, gaze and facial action units to CSV. The notebook then measures, per video and per action, whether a face was tracked and how confident the landmark detector was.

- **Faces are tracked in only 323 of 2,525 clips (12.8%).** UCF-101 is dominated by sports and full-body actions filmed from a distance, where faces are small, turned away or blurred.
- **Close-up, face-centred actions work.** Every ApplyLipstick and BrushingTeeth clip is tracked, followed by ApplyEyeMakeup, ShavingBeard and HeadMassage (80–92% of clips).
- **Landmark confidence is bimodal.** It has a large peak near zero and a smaller cluster above 0.8 (median 0.13, interquartile range 0.02–0.66). Many face tracks are marginal detections, so any downstream use of the action units should filter by confidence.
```
