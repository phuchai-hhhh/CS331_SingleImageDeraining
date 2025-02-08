# CS331 Single Image Deraining

## 👥 Group Members  

- Hồng Phúc Hải  
- Đinh Tuệ Đức  
- Huỳnh Đăng Khoa

---

## Introduction

This project focuses on removing rain streaks from a single image. It aims to enhance image quality by eliminating rain streak noise, which is essential for improving visual clarity in various applications such as surveillance, autonomous driving, and outdoor photography.

## Problem Statement

- **Input**: Images degraded by rain streaks.  
- **Output**: Restored images without rain streaks.  

### Dataset

The dataset consists of paired images:  
- **Xᵢ**: Clean image (original).  
- **Yᵢ**: Rain-streaked image, formulated as:  
  \[
  Yᵢ = Xᵢ + R
  \]
  where **R** represents rain streak noise.  

## Methods

This project explores three deep learning-based approaches for image deraining:

### 🔹 Transformer-based: **Restormer**  
- **Self-attention mechanism** to learn long-range dependencies between pixels.  
- **Multi-Head Attention** to capture different relationships within the image.  
- **Multi-Dconv Head Transposed Attention** to enhance global contextual representation.  
- **Gated-Dconv Feed-Forward Network** to refine spatial details.  

### 🔹 CNN-based: **PReNet**  
- **Recurrent structure** to iteratively refine image features.  
- **Residual learning** to preserve important image details.  
- **Multi-layer perception** for enhanced feature extraction.  

### 🔹 GAN-based: **Pix2Pix**  
- **Generator-discriminator framework** to reconstruct clean images from rain-streaked inputs.  
- **Adversarial loss** helps in generating realistic clean images.  

## Experiments

### Dataset

- **Dataset**: Artificially generated rain-streaked images with paired ground truth.  
- **Image resolution**: 256 × 256 pixels.  
- **Train-validation split**: 80% training, 20% validation.  

### Model Evaluation

**Metrics Used**:  
- **PSNR (Peak Signal-to-Noise Ratio)**: Measures image reconstruction quality.  
- **SSIM (Structural Similarity Index Measure)**: Evaluates perceptual similarity between restored and ground-truth images.  

**Experimental Settings**:  
- **Batch size**: {1, 4, 8, 16, 32}  
- **Learning rate**: 1e-4  
- **Optimizer**: Adam (`betas=(0.9, 0.999)`)  
- **Epochs**: 50  

## Results

| **Method**   | **Rain100H**      | **Rain100L**      | **Test100**       | **Test1200**      | **Test2800**      | **Average**        |
|-------------|------------------|------------------|------------------|------------------|------------------|------------------|
| **Restormer** | **28.56 / 0.8627** | **34.48 / 0.9567** | **27.91 / 0.8793** | **31.83 / 0.9133** | **32.87 / 0.9340** | **31.13 / 0.9092** |
| **PReNet**   | 23.303 / 0.7411  | 27.975 / 0.8727  | 22.861 / 0.8124  | 29.065 / 0.8831  | 30.477 / 0.9092  | 26.736 / 0.8437  |
| **Pix2Pix**  | 23.687 / 0.715   | 27.999 / 0.8748  | 23.481 / 0.8012  | 27.632 / 0.8474  | 28.560 / 0.8718  | 26.272 / 0.8220  |

## Demo

**Input Image**

<img src="https://github.com/phuchai-hhhh/CS331_SingleImageDeraining/blob/c663df4b41b43a1bc885fe84e7ca353f7e951dae/Input_image.jpg" width="600">

**Model Predictions**

<table>
  <tr>
    <th>Restormer Prediction</th>
    <th>Pix2Pix Prediction</th>
    <th>PReNet Prediction</th>
  </tr>
  <tr>
    <td><img src="https://github.com/phuchai-hhhh/CS331_SingleImageDeraining/blob/82db348ecea00af4a37b49bb69621623d985d081/Restormer_output.jpg" width="300"></td>
    <td><img src="https://github.com/phuchai-hhhh/CS331_SingleImageDeraining/blob/705a94de24755730a67dca6c0770b916665c8a5e/Pix2Pix_output.jpg" width="300"></td>
    <td><img src="https://github.com/phuchai-hhhh/CS331_SingleImageDeraining/blob/705a94de24755730a67dca6c0770b916665c8a5e/PReNet_output.jpg" width="300"></td>
  </tr>
</table>

## References

- **Restormer**: [paper]*Restormer: Efficient transformer for high-resolution image restoration.*  
- **PReNet**: [paper]*Progressive Image Deraining Networks: A Better and Simpler Baseline.*  
- **Pix2Pix**: [paper]*Image-to-Image Translation with Conditional Adversarial Networks.*
- **Dataset**: [paper]*Multi-Scale Progressive Fusion Network for Single Image Deraining.*  

## Contact

For inquiries regarding this project, please reach out via email or open an issue on this GitHub repository.
