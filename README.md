# Vision & Spatial AI Pipeline

This repository contains a comprehensive collection of computer vision projects. It demonstrates the ability to build, train, and deploy vision models across static images and complex video datasets, leveraging PyTorch for custom architectures and state-of-the-art frameworks for transfer learning. Hardware acceleration is optimized for Apple Silicon using the MPS backend.

## Core Deep Learning Projects (Static Images)

### 1. Image Captioning (Flickr8k)
*   **Modeling:** Evaluates a custom model against a pre-trained BLIP model using a custom `visualize_comparison_all_truths` function. 
*   **Architecture:** Explores replacing standard ResNet-50 backbones with modern Vision Transformers to extract richer visual features.

### 2. Image Colorization (Flowers102)
*   **Pipeline:** A comparative PyTorch study predicting RGB color channels from grayscale inputs.
*   **Data Handling:** Utilizes custom PyTorch `Dataset` and `DataLoader` classes to manage the Flowers102 dataset.

### 3. Object Detection (YOLO11n)
*   **Data Crawling:** Utilized `icrawler` and `beautifulsoup4` to programmatically scrape the web for background images to supplement the dataset.
*   **Transfer Learning:** Fine-tuned the Ultralytics YOLO11n model to detect custom classes (`waldo`, `wanda`, `woof`). 
*   **Results:** Achieved high precision and recall, tracking bounding box regressions and classification losses across a 10-epoch training cycle.

## Spatial & Real-Time Tracking (Video/Live Feed)

### 4. Hand Landmark Tracking
*   **Framework:** Built a live-feed analytics dashboard utilizing MediaPipe's `HandLandmarker` task. 
*   **UI Integration:** Features custom HTML overlays (`<div id="gesture_pill">`) to display dynamic performance metrics and gesture states alongside the video feed.

### 5. Facial Action Unit (AU) Extraction (UCF-101)
*   **Batch Processing:** Analyzes high-movement video datasets via OpenFace, iterating through the UCF-101 dataset for complex action classes like `BenchPress` and `TaiChi`. 
*   **Methodology:** Captures in-the-wild facial landmarks and Continuous Emotion (CECLM) parameters under varying environmental conditions.
