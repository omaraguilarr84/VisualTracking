# SERTnet: Segmentation and Eye-tracking in Real-time Network

**SERTnet** is a real-time, latency-aware eye segmentation framework optimized for deployment on resource-constrained hardware, such as mobile workstations and clinical devices. It is specifically designed for use in **gaze-contingent systems** in VR/AR, **ophthalmic screening**, and **robot-assisted surgery**, all of which demand **sub-20 ms end-to-end latency** without compromising accuracy.

---

## 🔍 Overview

Modern applications in medical imaging and extended reality require **fast and accurate** eye segmentation models. While existing solutions like **RITnet** deliver high accuracy, they are computationally intensive and poorly suited for real-time deployment.

**SERTnet** is a re-engineered version of RITnet that:
- Matches RITnet’s segmentation accuracy on the **openEDS 2019** dataset.
- Reduces model parameters by **37.9%**.
- Cuts inference time by **73.3%**.
- Achieves **95.9% mean IoU at over 85 Hz** on a **GeForce MX570 laptop**.

---

## ⚙️ Key Innovations

### ✅ Efficient Preprocessing (0.14 ms/frame)
- **Zero-copy LUT Pipeline**: Performs gamma correction and CLAHE on the CPU for minimal overhead.

### 🧠 Network Optimizations
- **MobileNetV2-style Inverted Residuals**: Replaces dense blocks for better performance on mobile GPUs.
- **Depthwise Separable Convolutions**: Reduces computation and memory footprint.

### ✂️ Model Compression
- **Structured Pruning**: Targets redundant filters while preserving key features.
- **8-bit Quantization**: Achieves significant latency gains with minimal accuracy drop.

---

## 🧪 Performance

| Metric               | RITnet     | SERTnet    |
|----------------------|------------|------------|
| mIoU (%)             | 95.9       | 95.9       |
| Inference Speed (Hz) | ~49        | **85+**    |
| Model Params         | Baseline   | **-37.9%** |
| Inference Latency    | > 20 ms    | **< 6 ms** |

Tested on: **GeForce MX570**, Laptop-class GPU

---

## 📂 Dataset

We evaluate on the **openEDS 2019** dataset (public eye segmentation dataset by Facebook Reality Labs):
- [openEDS Dataset](https://research.fb.com/publications/openeds-a-structured-dataset-and-baseline-evaluations-for-eye-segmentation-in-vr/)

---
