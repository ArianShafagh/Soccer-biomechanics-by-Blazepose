# 🏃‍♂️ Soccer Biomechanics by BlazePose

> **A high-performance biomechanical analysis pipeline using MediaPipe BlazePose for posture estimation, joint kinematics, symmetry evaluation, and alignment assessment**

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python)
![MediaPipe](https://img.shields.io/badge/MediaPipe-BlazePose-orange?logo=google)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv)

**Author:** [Arian Shafagh](https://github.com/ArianShafagh)
**Affiliation:** FCRLAB, University of Messina (UNIME)
**Project:** Internship – Module 2 (Biomechanical Analysis Pipeline)

</div>

---

## 📋 Table of Contents

* [Overview](#-overview)
* [Key Features](#-key-features)
* [Installation](#-installation)
* [Usage](#-usage)
* [Pipeline Workflow](#-pipeline-workflow)
* [Biomechanical Metrics](#-biomechanical-metrics)
* [Outputs](#-outputs)
* [Project Structure](#-project-structure)
* [Requirements](#-requirements)
* [References](#-references)

---

## 🔍 Overview

This project implements a **complete biomechanical analysis pipeline** for human motion using **MediaPipe BlazePose**.

After evaluating multiple pose estimation models (LightPose, MoveNet, OpenPose, HRNet), BlazePose was selected due to its:

* ✅ **33-point 3D landmark system**
* ⚡ **Real-time performance**
* 🎯 **High accuracy for biomechanical applications**

The system processes input videos to extract posture data and compute **quantitative biomechanical metrics**, making it suitable for:

* ⚽ Sports performance analysis
* 🏥 Clinical posture assessment
* 🧪 Research in human motion

> 🎯 **Objective:** Build a reliable, frame-independent biomechanical analysis system using robust 3D pose estimation.

---

## ✨ Key Features

* 🧍 **33 Landmark Tracking** (full-body skeleton)
* 📐 **Joint Angle Computation**

  * Elbow, Knee, Shoulder, Hip, Ankle
* ⚖️ **Symmetry Analysis**

  * Left vs Right body comparison
* 🧭 **Alignment Metrics**

  * Trunk forward/backward lean
  * Trunk lateral lean
  * Knee valgus (collapse detection)
* 🎥 **Annotated Video Output**
* 📊 **Structured JSON Outputs**
* ⚙️ **Configurable Pipeline (via `config.json`)**
* 🧪 Designed for **evaluation & validation tasks**
* ❗ BlazePose has **3 different modes for evaluation** which are **lite** and **full** and **heavy** (lite version is the fastest and heavy is most accurate and full is the moderate) 

---

## ⚙️ Installation

### 1. Clone Repository

```bash
git clone https://github.com/ArianShafagh/Soccer-biomechanics-by-Blazepose.git
cd Soccer-biomechanics-by-Blazepose
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv
```

**Activate:**

* Windows:

```bash
venv\Scripts\activate
```

* Linux/Mac:

```bash
source venv/bin/activate
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### 1. Configure Input (`config.json`)

```json
{
    "model_path": ".\\BlazeModels\\pose_landmarker_lite.task",  
    "video_path": ".\\Video\\soccerTest.mp4",
    "output_segmentation_masks": false,
    "min_pose_detection_confidence": 0.5,
    "min_tracking_confidence": 0.5,
    "min_pose_presence_confidence": 0.5,
    "start_frame": 0,
    "end_frame": null,
    "show_biomechanical_data": true,
    "video_output": true
}
```

---

### 2. Run the Pipeline

```bash
py main.py
```

---

### 🎮 Controls (During Execution)

* `Space` → Pause / Resume
* `q` → Quit and save outputs

---

## 🔄 Pipeline Workflow

```mermaid
graph LR
    A[Input Video] --> B[Frame Extraction]
    B --> C[BlazePose Detection]
    C --> D[Visibility Filtering]
    D --> E[Coordinate Processing]
    E --> F[Biomechanics Computation]
    F --> G[Output Generation]
```

---

### 🧠 Processing Strategy

* 🎨 **Image Landmarks (x,y,z normalized)** → visualization only
* 🌍 **World Landmarks (meters)** → biomechanical computation
* ⚠️ **Z-axis** → used only for alignment (reduces noise in angles)

---

## 📐 Biomechanical Metrics

### 1. Joint Angles

```math
θ = arccos( (BA · BC) / (||BA|| ||BC||) )
```

* Computed for:

  * Elbow, Knee, Shoulder, Hip, Ankle

---

### 2. Symmetry

```math
Symmetry = |Left - Right|
```

* Lower values → better balance
* Applied to all major joints

---

### 3. Alignment

#### 🧍 Trunk Inclination

* Based on shoulder–hip midpoint axis
* Uses `atan2` for robust orientation

Outputs:

* Forward/Backward lean
* Side lean

---

#### 🦵 Knee Valgus Proxy

```text
Valgus = 180° - Knee Angle
```

* `0°` → neutral
* `>0°` → inward collapse

---

## 📁 Outputs

### 🎬 Annotated Video

* Skeleton overlay
* Optional real-time metrics

---

### 📍 Landmark JSON

| File                   | Description               |
| ---------------------- | ------------------------- |
| `landmarks_pixel.json` | Image coordinates         |
| `landmarks_world.json` | Real-world 3D coordinates |

---

### 📊 Biomechanical JSON

Includes:

* Angles
* Symmetry
* Alignment

---

## 🗂️ Project Structure

```
Soccer-biomechanics-by-Blazepose/
│
├── main.py
├── model.py
├── biomechanics.py
├── loader.py
├── config.json
├── requirements.txt
│
├── BlazeModels/
│   ├── pose_landmarker_lite.task
│   ├── pose_landmarker_full.task
│   └── pose_landmarker_heavy.task
│
├── output/
│   ├── annotated_video.mp4
│   ├── landmarks_pixel.json
│   ├── landmarks_world.json
│   └── biomechanics.json
│
└── venv/
```

---

## 📦 Requirements

```txt
mediapipe>=0.10.32
opencv-python>=4.x
numpy
```

---

## 📚 References

* MediaPipe BlazePose Documentation
* BlazePose Paper (Bazarevsky et al., 2020)
* OpenCV Documentation
* Physio-pedia (Knee Valgus)

---

## 🔮 Future Improvements

* Multi-person tracking
* Real-time webcam support
* GUI dashboard
* Injury risk prediction models

---

## ⭐ Final Note

If you found this project useful, consider giving it a ⭐ on GitHub!

---