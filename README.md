# Siamese Neural Network for Audio Deepfake Detection

## Overview
This project presents a refined implementation of a Siamese Convolutional Neural Network (SCNN) for detecting audio deepfakes using the DECRO dataset. The model architecture is designed to effectively learn discriminative audio embeddings that differentiate between genuine and fake speech samples. It leverages contrastive learning principles in a supervised setup to optimize similarity-based verification.

---

## Technical Description
The model is based on a Siamese CNN architecture, which is particularly well-suited for verification tasks that depend on similarity measurement rather than categorical classification. The SCNN works by passing two inputs through identical convolutional subnetworks to obtain embeddings that are compared using a distance metric.

The core of the architecture includes residual blocks that allow the training of deep models by preserving feature integrity across layers, and Squeeze-and-Excitation (SE) blocks that help highlight meaningful features through channel-wise recalibration. The network outputs 256-dimensional embeddings used to compute Euclidean distances, which form the basis for contrastive loss optimization.

**Our Implementation:**
1. Incorporated three stages of residual blocks followed by an SE block for refined feature recalibration.
2. Added dynamic embedding projection layers initialized during the first forward pass to accommodate flexible model depth.

---

## Dataset
The training and evaluation phases use the **DECRO (Deepfake Corpus for Robustness and Optimization)** dataset, consisting of paired audio samples and similarity labels (0: same speaker, 1: different). 

**Structure:**
- Training list file: `en_dev.txt`, audio path: `inner/wav/en_dev/`
- Evaluation list file: `en_eval.txt`, audio path: `inner/wav/en_eval/`

**Disclaimer**: The dataset is **not** included in this repository. It can be downloaded from the official GitHub page:
[https://github.com/petrichorwq/DECRO-dataset](https://github.com/petrichorwq/DECRO-dataset)

---

## Training Configuration
- **Batch Size**: 32
- **Epochs**: 40
- **Optimizer**: Adam
- **Learning Rate**: 0.0005
- **Contrastive Loss Margin**: 2.0
- **Input Format**: 1-channel spectrograms (128x128)
- **Embedding Size**: 256

---

## Evaluation and Threshold Optimization
To tune decision sensitivity, the model is validated against varying thresholds ranging from 0.20 to 2.10 in increments of 0.05. For each threshold, prediction accuracy is calculated on the evaluation dataset. This process helps identify the threshold at which the similarity score most effectively separates similar and dissimilar audio pairs.

The best-performing threshold, along with its accuracy, is reported as part of the validation step. This threshold becomes the benchmark for classifying unknown inputs.

---

## Author's Recommendations
- **Input Consistency**: Ensure all inputs are preprocessed into fixed-size, normalized spectrograms.
- **Threshold Sweeping**: Always conduct post-training threshold evaluation between 0.2 and 2.1 (step size: 0.05) to optimize classification performance.
- **Data Augmentation**: Introduce variations like pitch shifts, noise injection, or time warping to improve generalization.
- **Checkpointing**: Regularly save model states during long training runs to avoid data loss.
- **GPU Utilization**: Leverage CUDA acceleration for faster training.
- **Visual Diagnostics**: Use dimensionality reduction tools (e.g., t-SNE, PCA) to visualize embedding separability.
- **Batch Tuning**: Adjust batch size based on available memory and hardware constraints for stable training.

---

## Python Libraries and Packages Used
The following Python libraries were employed throughout the project:
- `numpy`: Core numerical operations
- `pandas`: Dataframe manipulation and I/O
- `librosa`: Audio signal processing and spectrogram computation
- `scikit-learn`: Evaluation metrics and data splitting utilities
- `torch`: Deep learning modeling and training framework
- `torchvision`: Tensor transformations
- `torchaudio`: PyTorch-compatible audio loading and processing
- `tqdm`: Training and evaluation progress tracking
- `matplotlib`: Plotting for result visualization
- `seaborn`: Enhanced statistical plotting interface

---

## Author
**Kartik Dua**  
*Dated: 05/04/2025*

