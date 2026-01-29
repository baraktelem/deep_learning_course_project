# Wavelet Scattering Transform (WST) Integration with CNNs
### Deep Learning Course Project

![Project Status](https://img.shields.io/badge/Status-Active-green)
![Python](https://img.shields.io/badge/Python-3.x-blue)
![Framework](https://img.shields.io/badge/PyTorch-Deep%20Learning-orange)

## 📌 Background
In this project, we examine the integration of **Wavelet Scattering Transform (WST)** kernels into Convolutional Neural Network (CNN) architectures. We explore two primary hypotheses:

1. **Texture Offloading:** By integrating fixed wavelet kernels into early layers, we can "offload" the texture detection task to the WST. This frees the subsequent learnable kernels to focus on higher-level semantic tasks more efficiently.
2. **Scattering as a Prior:** WST imposes a strong geometric prior (translation invariance and deformation stability) on the network. We hypothesize that this prior is particularly beneficial in **low-data regimes**, where standard CNNs struggle to learn effective filters from scratch without overfitting.

## 📂 Repository Structure
The repository is organized as follows:

```text
├── artifacts/      # Stores training outputs, saved models (.pth), and logs
├── data/           # Dataset storage (CIFAR10, etc.) and preprocessing scripts
├── docs/           # Documentation and presentations
├── environment/    # Configuration files for setting up the environment (Conda)
├── notebooks/      # Jupyter Notebooks for all experiments (Baselines & WST)
├── src/            # Source code for models, scattering layers, and utility functions
│   ├── models/
│   │   ├── architectures/  # Model architectures (ResNet18, ScatNet18, HybridGaborResNet18)
│   │   └── layers/         # Custom layers (Scattering, Gabor, Hybrid)
│   └── utils/              # Training, configuration, datasets, visualization utilities
├── stats/          # CSVs and logs containing experiment results and statistics
└── README.md
```

## 📓 Notebooks Overview

| File | Description |
|------|-------------|
| `baseline.ipynb` | CIFAR10 ResNet18 Baseline (Full Dataset - 4000 samples/class) |
| `baseline-acc-vs-n-samples_10_50_100_1000_4000.ipynb` | ResNet18 performance across multiple data regimes (10, 50, 100, 1000, 4000 samples/class) |
| `baseline-acc-vs-n-samples_200_300_400.ipynb` | ResNet18 performance for intermediate sample sizes (200, 300, 400 samples/class) |
| `scatresnet_by_L.ipynb` | ScatResNet18 with L parameter search (varying scattering channels) |
| `scatresnet_acc_vs_n_samples.ipynb` | ScatResNet18 performance across different data regimes |
| `models_robustness.ipynb` | Variance/Stability analysis of model architectures |
| `hybridgabor-acc-vs-n-samples.ipynb` | HybridGabor architecture experiments across data regimes |
| `maxhybridgabor_acc_layers_vs_n_samples.ipynb` | MaxHybridGabor layer depth analysis across sample sizes |
| `maxhybridgabor_acc_layers_vs_n_samples_4000_samples.ipynb` | MaxHybridGabor layer analysis on full dataset |
| `maxhybridgabor-acc-layers-vs-n-samples_L1-2_L1-3.ipynb` | MaxHybridGabor with specific L configurations (L1-2, L1-3) |
| `maxhybridgabor-acc-layers-vs-n-samples_L1-4.ipynb` | MaxHybridGabor with L1-4 configuration |
| `maxhybridgabor-acc-layers-vs-n-samples_L1-5.ipynb` | MaxHybridGabor with L1-5 configuration |
| `maxhybridgabor-acc-vs-n-samples.ipynb` | MaxHybridGabor performance analysis across data regimes |

## 🛠️ Prerequisites

To run the experiments, you need the following dependencies installed:

- **Python 3.x**
- **PyTorch** (and torchvision)
- **Kymatio** (Required for Wavelet Scattering implementations)
- **Jupyter Notebook**
- **NumPy**, **Matplotlib** (for data processing and visualization)

## 🚀 How to Run

### Option 1: Running Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/baraktelem/deep_learning_course_project.git
   cd deep_learning_course_project
   ```

2. **Set up the environment:**
   ```bash
   # Using Conda (recommended)
   conda env create -f environment/environment.yaml
   conda activate deep_learning_project
   
   # Or using pip
   pip install torch torchvision kymatio jupyter numpy matplotlib
   ```

3. **Run the experiments:**
   Launch Jupyter and open any notebook in the `notebooks/` folder:
   ```bash
   jupyter notebook
   ```

### Option 2: Running on Kaggle

To run these notebooks on Kaggle with the correct directory structure:

1. **Create a new Notebook** on Kaggle.
2. **Copy the content** of the desired notebook (e.g., `notebooks/scatresnet_by_L.ipynb`) into the Kaggle editor.
3. **Add the Repository as a Dataset:**
   - Go to the "Data" panel in the sidebar.
   - Click "Add Data" → "GitHub".
   - Paste the repository URL: `https://github.com/baraktelem/deep_learning_course_project`
   - Set the dataset name as: `deep_learning_course_project`

## 📊 Method: Hybrid Layer Designs

We developed a set of **Hybrid** layers that combine standard learnable convolutions with analytic frequency-based filters (Gabor wavelets). These layers were integrated into a ResNet18 backbone, replacing standard convolutional blocks at various depths to investigate their impact on sample efficiency and robustness.

### 1. Fixed Scattering (Channel-wise Integration)

Our initial approach utilized the **Wavelet Scattering Transform** to inject fixed, translation-invariant features.

**Mechanism:**

1. **Channel Budget Split:** The layer divides the total channel budget between standard convolutions and scattering channels.
2. **Feature Generation:** We generate fixed texture descriptors using a Scattering Transform. The scattering path uses:
   - Wavelet convolutions followed by modulus nonlinearity
   - Average pooling for stability
3. **Resolution Matching:** Since the scattering transform downsamples the input, we apply a **Transposed Convolution** layer (stride 2, kernel size 2) to upsample the features back to the required output spatial resolution ($32 \times 32$).
4. **Concatenation:** The upsampled scattering maps are concatenated with the standard learned features from the convolutional path.

**Key Properties:**
- Provides strong mathematical guarantees (local translation invariance, stability to deformations)
- Fixed filters do not require learning
- Acts as structural regularization

### 2. Parametric Hybrid (Channel-wise & Element-wise)

While the fixed scattering transform offers strong guarantees, it imposes a rigid "one-size-fits-all" filter bank. To bridge this gap, we developed a **Parametric Hybrid Layer** that retains the Gabor structure but makes the filter parameters learnable.

**Mechanism:**

- **Learnable Parameters:** Instead of storing static weight matrices, the layer learns the generative parameters of Gabor functions:
  - **Orientation** ($\theta$): Direction of the filter
  - **Scale** ($\sigma$): Size of the Gaussian envelope
  - **Wavelength** ($\lambda$): Frequency of the sinusoidal component
- **On-the-Fly Generation:** During each forward pass, Gabor filters are generated dynamically from these parameters
- **Optimization Strategy:**
  - Standard gradient descent with backpropagation through the generation step
  - Higher learning rate for geometric parameters (no weight decay)
  - Initialization: Uniform distribution across orientations and frequencies
- **Integration:** Similar structure to fixed scattering with modulus nonlinearity and average pooling

**Advantages:**
- Adapts the biological prior to specific dataset statistics
- Moves from static "hard-coded" prior to data-adaptive one
- Balances inductive bias with flexibility

### 3. Integration Strategies

We investigated two distinct fusion strategies for combining Gabor and standard convolutional features:

#### Channel-wise Integration (Concatenation)

- **Channel Split:** Total channel budget is divided between standard and Gabor paths
- **Pooling:** Stride-1 average pooling to preserve spatial resolution (mimicking scattering stability)
- **Fusion:** Outputs are concatenated channel-wise
- **Effect:** Replaces layers without increasing network width (parameter-efficient)

#### Element-wise Integration (Competitive Max-Out)

- **Full Capacity:** Both paths maintain full channel count
- **DoG Attention:** The Gabor path includes a Difference of Gaussians (DoG) attention gate for spatial attention
- **Fusion:** Element-wise maximum between paths - forces competition
- **Effect:** Network dynamically "selects" the biological prior only when it produces stronger responses than learned filters

## 🏗️ Architectures

We utilized **ResNet18** as our backbone, systematically introducing hybrid layers to evaluate their impact.

### Architecture Variants

#### Baseline: CIFAR10 Adapted ResNet18

We adapted the standard ResNet18 for CIFAR10's smaller input size ($32\times32$).

**Architecture Adaptations:**

| Layer Name | Original Output Size | Original Configuration | Adapted Output Size | Adapted Configuration |
| :--- | :--- | :--- | :--- | :--- |
| **Conv1** | $112 \times 112$ | $7 \times 7$ Conv, 64 filters, stride 2 | $32\times 32$ | $3\times3$ Conv, 64 filters, stride 1 |
| **Pooling** | $56 \times 56$ | $3 \times 3$ Max Pooling, stride 2 | - | Removed |
| **Conv2_x** | $56 \times 56$ | [3x3, 64] x 2 blocks | $16 \times 16$ | Same |
| **Conv3_x** | $28 \times 28$ | [3x3, 128] x 2 blocks | $8 \times 8$ | Same |
| **Conv4_x** | $14 \times 14$ | [3x3, 256] x 2 blocks | $4 \times 4$ | Same |
| **Conv5_x** | $7 \times 7$ | [3x3, 512] x 2 blocks | $2 \times 2$ | Same |
| **Output** | $1 \times 1$ | Avg Pool $\rightarrow$ 1000-d FC $\rightarrow$ Softmax | $1 \times 1$ | Avg Pool $\rightarrow$ 10-d FC $\rightarrow$ Softmax |

**Model Statistics:**
- **Total Parameters:** 11,173,962
- **Model Size:** 42.63 MB

#### Layer 1 Replacement Architectures

The first convolutional layer (Conv1) extracts initial low-level features. We replaced this single layer with our three hybrid variants:

1. **ScatResNet:** Replaces Conv1 with **Fixed Scattering Hybrid** (Concatenation)
2. **HybridResNet:** Replaces Conv1 with **Parametric Hybrid** (Concatenation)
3. **MaxHybridResNet:** Replaces Conv1 with **Parametric Hybrid** (Element-wise Maximum)

#### Deep Integration Architectures (Progressive Replacement)

Based on initial experiments, we selected the **MaxHybrid** architecture for deeper integration testing. We defined a progression where **L** represents the number of consecutive convolutional layers replaced:

- **L=1:** Only Conv1 replaced (stem layer)
- **L=1-3:** Conv1 + first residual block (Conv2_x) replaced
- **L=1-5:** Conv1 + Conv2_x + Conv3_x replaced

This progression allows us to determine if Gabor priors remain beneficial as feature abstraction increases, or if their utility is confined to initial pixel-level processing.

## 🧪 Experiments and Results

### Experimental Setup

**Dataset:** CIFAR-10 (10 classes, 32×32 RGB images)

**Training Configuration:**
- **Epochs:** 200
- **Batch Size:** 128
- **Optimizer:** SGD with momentum 0.9, weight decay $5\times10^{-4}$
- **Learning Rate:** Initial 0.1 with Cosine Annealing scheduler ($T_{max} = 200$)
- **Data Augmentation:** Standard augmentations (random crops, horizontal flips)

**Evaluation Strategy:**
- Models trained on stratified subsets: 10, 50, 100, 200, 300, 400, 500, 1000, 4000 samples per class
- Validation: 5000 samples
- Test: Full CIFAR-10 test set (10,000 samples)

### Experiment 1: First Layer Integration

**Objective:** Evaluate the impact of replacing the initial convolutional layer (Conv1) with three hybrid variants across different data regimes.

**Models Compared:**
- Baseline ResNet18
- ScatResNet (Fixed Scattering)
- HybridResNet (Parametric Concatenation)
- MaxHybridResNet (Parametric Max-Out)

#### Results: Sample Efficiency Across Data Regimes

**Low-Data Regime (< 200 samples per class):**

All hybrid architectures **significantly outperformed** the baseline:

| Samples/Class | Baseline | ScatResNet | HybridResNet | MaxHybridResNet |
|:---|:---|:---|:---|:---|
| 10 | ~15% | ~25% | ~25% | ~30% |
| 50 | ~40% | ~48% | ~50% | ~50% |
| 100 | ~46% | ~58% | ~58% | ~55% |

**Analysis:** The baseline achieves high training accuracy but poor test accuracy, indicating rapid overfitting. The hybrid layers impose structural priors that restrict the solution space, acting as strong architectural regularizers and preventing memorization of noise.

**Middle Regime (200-500 samples per class):**

Hybrid variants performed similarly, with **MaxHybridResNet** showing a slight advantage:

| Samples/Class | Baseline | ScatResNet | HybridResNet | MaxHybridResNet |
|:---|:---|:---|:---|:---|
| 200 | ~56% | ~65% | ~67% | ~68% |
| 300 | ~68% | ~72% | ~70% | ~72% |
| 500 | ~74% | ~74% | ~73% | ~75% |

**Analysis:** The competitive max-out mechanism offers a more robust feature selection strategy. When data is sufficient to learn basic patterns but insufficient for robust feature detectors, the max operation allows the network to "fall back" on stable Gabor features whenever learned filters are weak or noisy.

**High-Data Regime (≥ 1000 samples per class):**

The baseline **surpassed** hybrid models, though the gap remained minimal for max-out and fixed variants:

| Samples/Class | Baseline | ScatResNet | HybridResNet | MaxHybridResNet |
|:---|:---|:---|:---|:---|
| 1000 | ~86% | ~85% | ~83% | ~85% |
| 4000 | 94.14% | 93.46% | 92.80% | 93.97% |

**Analysis:** With sufficient data, standard CNNs can learn filters more specifically adapted to the dataset than fixed general wavelet kernels. Notably, the **Parametric Concatenation** (HybridResNet) underperformed the fixed scattering baseline - while the fixed transform guarantees signal preservation, learning parameters can disrupt these properties. This reflects optimization difficulty of geometric parameters and potential limitations in the generator design.

**Key Finding:** The element-wise max-out strategy provides the best balance, maintaining competitive performance across all data regimes.

### Experiment 2: Deep Integration

**Objective:** Determine if structural priors remain beneficial when integrated deeper into the network.

**Models Compared:**
- Baseline ResNet18
- MaxHybridResNet L=1 (Conv1 only)
- MaxHybridResNet L=1-3 (Conv1 + Conv2_x)
- MaxHybridResNet L=1-5 (Conv1 + Conv2_x + Conv3_x)

**Rationale for Max-Out Selection:** The competitive gating mechanism preserves full capacity of the standard path while allowing Gabor priors to override learned features when biological signals are stronger. Unlike concatenation, this approach doesn't reduce learnable parameter space in critical early layers.

#### Results: Impact of Depth on Sample Efficiency

| Samples/Class | Baseline | L=1 | L=1-3 | L=1-5 |
|:---|:---|:---|:---|:---|
| 100 | ~46% | ~55% | ~58% | ~58% |
| 200 | ~56% | ~67% | ~68% | ~68% |
| 500 | ~74% | ~75% | ~74% | ~74% |
| 1000 | ~86% | ~85% | ~86% | ~87% |

**Key Observations:**

1. **Diminishing Returns:** Adding priors to deeper layers improved performance over baseline, but marginal gains decreased compared to the initial L=1 impact.

2. **L=1-3 Sweet Spot:** Replacing the stem and first residual block (L=1-3) performed slightly better than single-layer replacement (L=1) across almost all data regimes, with the exception of 500 samples/class where performance was comparable.

3. **Deepest Integration (L=1-5):** Showed slight improvement in the 1000 samples/class regime, suggesting deep structural priors may aid generalization with moderate data availability.

4. **Computational Cost:** Training time for L=1-5 was approximately **5× slower** than baseline due to on-the-fly filter generation in deeper, wider blocks. This made evaluation on larger datasets (>1000 samples/class) prohibitive.

**Conclusion:** Structural priors are most effective in early network stages (L=1 to 3), confirming that analytic priors are best suited for early visual processing rather than deep semantic abstraction.

### Experiment 3: Robustness Analysis

**Objective:** Evaluate generalization capabilities by testing robustness against texture perturbations and geometric transformations.

**Models Tested:**
1. ResNet18 (Baseline)
2. ScatResNet18 (Fixed Scattering)
3. MaxHybridGaborLayer2 (L=1-2, Parametric)
4. MaxHybridGaborLayer3 (L=1-3, Parametric)

All models trained on full CIFAR-10 (4000 samples/class).

#### Augmentation Categories

Using the **Kornia** library, we applied six augmentations organized into three subgroups:

1. **Positional Changes:** Horizontal Flip, Vertical Flip
2. **Color Perturbations:** Grayscale, Gaussian Noise
3. **Blur Operations:** Gaussian Blur, Median Blur

#### Results: Classification Accuracy on Augmented Test Set

**Low-Data Regime Performance (< 500 samples/class):**

Hybrid models (both fixed and learnable scattering) generally outperform baseline ResNet across most augmentation types.

**Notable Exceptions:**
- **Gaussian Noise:** Fixed scattering performs significantly worse, but learnable scattering variants perform better across all data sizes
- **Median Blur:** Fixed scattering performs better than all variants across all data sizes

#### Results: Accuracy Drop (Robustness Metric)

We measured robustness as the accuracy drop relative to clean test performance.

**Positional Augmentations (Flips):**
- All models exhibit comparable degradation magnitude
- No clear winner due to high noise in performance curves

**High-Data Regime Analysis (4000 samples/class):**

| Augmentation | Baseline Drop | ScatResNet Drop | Interpretation |
|:---|:---|:---|:---|
| **Grayscale** | ~5% | ~7% | ScatResNet more sensitive |
| **Gaussian Noise** | ~20% | ~25% | ScatResNet more sensitive |
| **Gaussian Blur** | ~15% | ~13% | ScatResNet more robust |
| **Median Blur** | ~12% | ~6% | ScatResNet more robust |

**Key Findings:**

1. **Noise Sensitivity:** Fixed scattering is significantly more sensitive to Grayscale and Gaussian Noise than ResNet baseline.
   - **Hypothesis:** Noise acts as high-frequency corruption, triggering spurious high activations in scattering channels that disrupt classification.

2. **Blur Robustness:** ScatResNet considerably outperforms baseline on both Gaussian and Median Blur.
   - **Hypothesis:** Blur functions as a low-pass filter, suppressing high frequencies. This forces the classification head to bypass inactive scattering coefficients and rely on learnable CNN kernels processing remaining low-frequency information.
   - **Evidence:** Accuracy drop is notably lower for Median Blur (6-8%) vs Gaussian Blur (8-14%). Median Blur preserves edge structures (valid high-frequency components) better, allowing scattering transform to leverage retained information.

3. **Validation of Theory:** These findings validate the texture offloading hypothesis - by delegating texture detection to scattering layers, CNN kernels are liberated to learn complementary structural features.

## 📈 Summary of Results

### Main Findings

1. **Sample Efficiency:** Hybrid models with Gabor priors significantly outperform baselines in low-data regimes (<500 samples/class), with improvements of **15-30%** in test accuracy.

2. **Data Regime Behavior:**
   - **Low Data (< 200):** Hybrid models excel - structural priors prevent overfitting
   - **Middle Data (200-500):** Max-out variant shows slight advantage
   - **High Data (≥ 1000):** Baseline surpasses hybrids - learned filters become more dataset-specific

3. **Architecture Comparison:**
   - **Fixed Scattering:** Best robustness to blur, but sensitive to noise
   - **Parametric Max-Out:** Best overall balance across all data regimes
   - **Parametric Concatenation:** Underperforms due to optimization challenges

4. **Depth Dependence:** Structural priors most effective in early stages (L=1-3). Deeper integration (L=5) shows diminishing returns with 5× computational cost.

5. **Robustness Trade-off:**
   - **Superior:** Blur robustness (Gaussian, Median)
   - **Inferior:** Noise sensitivity (Gaussian noise, Grayscale)
   - **Interpretation:** Offloading texture to wavelets encourages learned kernels to focus on robust, low-frequency structural features

6. **Transfer Learning:** ScatResNet converges 1.4-1.5× faster when transferring to new tasks, validating improved adaptability.

### Practical Recommendations

**Use Hybrid Architectures When:**
- Working with limited labeled data (<500 samples/class)
- Transfer learning scenarios requiring fast adaptation
- Applications where blur robustness is critical
- Resource constraints favor parameter-efficient models

**Use Standard CNNs When:**
- Large datasets available (>1000 samples/class)
- Computational efficiency is paramount
- Tasks require robustness to high-frequency noise

## 🔮 Future Work

### 1. Transfer Learning

The observed performance gains in the low-data regime suggest that scattering integration enhances the network's adaptability. To empirically validate this, we conducted a **preliminary transfer learning experiment** as a proof of concept.

**Experimental Design:**
- **Pre-training:** Both models (Baseline ResNet18 and ScatResNet18) were fully trained on an initial 5-class subset of CIFAR-10 (Classes 0-4)
- **Fine-tuning:** Models were then fine-tuned on a distinct set of 5 classes (Classes 5-9)
- **Metric:** Number of epochs required to reach specific accuracy milestones on the new task

**Key Finding:** The fixed scattering model (ScatNet) converges faster to the new data, reaching 90% accuracy in fewer epochs than the baseline. This suggests that the fixed scattering features provide a strong foundation that accelerates adaptation to new visual categories without needing to relearn fundamental edge detectors.

**Future Directions:**
- Test transfer learning across different domains (e.g., CIFAR-10 → STL-10, natural images → medical imaging)
- Evaluate transfer at different pre-training data sizes
- Compare transfer learning efficiency with other architectural regularization techniques

### 2. Mitigating Texture Bias

Our robustness experiments revealed that scattering transforms may inadvertently induce strong texture bias, evidenced by disproportionate accuracy drops under high-frequency corruption (noise, grayscale).

**Proposed Solutions:**
- **Scattering Channel Regularization:** Apply L1/L2 penalties to scattering activations to prevent over-reliance
- **Dropout on Scattering Features:** Randomly suppress scattering channels during training
- **Attention Mechanisms:** Learn to dynamically weight scattering vs. standard features

**Goal:** Encourage more balanced representations integrating both textural and structural features.

### 3. Optimization of Parametric Gabor

The parametric concatenation variant underperformed fixed scattering in high-data regimes, suggesting optimization challenges.

**Areas for Improvement:**
- **Advanced Initialization Schemes:** Better parameter initialization strategies
- **Architectural Search:** Optimize generator design for on-the-fly filter creation
- **Regularization:** Explore constraints that preserve scattering transform properties during learning

### 4. Scaling to Higher Resolutions

Current experiments used CIFAR-10 (32×32 resolution) due to computational constraints (Kaggle resources).

**Next Steps:**
- Test on ImageNet (224×224) or other high-resolution datasets
- Investigate if findings generalize beyond small-scale images
- Explore hierarchical scattering at multiple scales

### 5. Domain-Specific Applications

**Medical Imaging:** Where labeled data is extremely scarce and texture patterns are diagnostically relevant

**Remote Sensing:** Satellite imagery with limited ground truth labels

**Manufacturing Defect Detection:** Specialized applications with small datasets

## 👥 Contributors

- **Barak Telem** - [GitHub](https://github.com/baraktelem)
- **Gilad Navok** - [GitHub](https://github.com/giladnavok)

**Course:** Deep Learning 00460217, Technion - Israel Institute of Technology

## 📚 References

1. Bruna, J., & Mallat, S. (2013). Invariant Scattering Convolution Networks. *IEEE Transactions on Pattern Analysis and Machine Intelligence*.
2. Gauthier, B., et al. Parametric Scattering Networks.
3. Geirhos, R., et al. ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness.
4. Andreux, M., et al. Kymatio: Scattering Transforms in Python.
