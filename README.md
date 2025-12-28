# deep_learning_course_project
In our project we plan to examine the integration of wavelet kernels into CNN architectures. By integrating wavelets kernels to CNN layers, we aim to offload the texture detection task to the wavelet kernels, freeing the standard kernels to perform other tasks.

# Baselines
We trained 4 variations of two baseline models.
## CIFAR10 Adapted ResNet18 Model
### Adaptions
| Layer Name | Output Size | Original Configuration | Adapted Configuration |
| :--- | :--- | :--- | :--- |
| **Conv1** | $112 \times 112$ | $7 \times 7$ Convolution, 64 filters, stride 2 | $3\times3$ Convolution, 64 filters, stride 1 |
| **Pooling** | $56 \times 56$ | $3 \times 3$ Max Pooling, stride 2 | Removed |
| **Conv2_x** | $56 \times 56$ | `[3x3, 64; 3x3, 64] x 2`| Same |
| **Conv3_x** | $28 \times 28$ | `[3x3, 128; 3x3, 128] x 2`| Same | 
| **Conv4_x** | $14 \times 14$ |`[3x3, 256; 3x3, 256] x 2` | Same |
| **Conv5_x** | $7 \times 7$ |`[3x3, 512; 3x3, 512] x 2` | Same |
| **Output** | $1 \times 1$ | Average Pool $\rightarrow$ 1000-d FC $\rightarrow$ Softmax | Average Pool $\rightarrow$ 10-d FC $\rightarrow$ Softmax

**Total Parameters:** 11,173,962
**Model Size:** 42.63 MB

### Trained versions
All versions were trained on the CIFAR10 train dataset for 200 epochs and evaluated on the CIFAR10 test dataset with basic augmentation.

- *baseline.ipynb*: Trained on 4000 samples for each class.
- *baseline-500.ipynb*: Trained on 500 samples for each class.
- *baseline-100.ipynb*: Trained on 100 samples for each class.

#### Results
|Statistic| baseline | baseline-500 | baseline-100 |
| :--- | :--- | :--- | :---|
| Test Accuracy | 94.14% | 77.02% | 46.10% |
| Train Accuracy | Fill | Fill | Fill |
| Training Loss | 0.004 | 0.01 | 0.027 |
| Training Time | 100m | 18m | 7m |


## CSIRO Adapted ResNet18 Model
This model is identical to the original ResNet18 but with 5 dimentional output.

### Adaptions
| Layer Name | Output Size | Original Configuration | Adapted Configuration |
| :--- | :--- | :--- | :--- |
| **Conv1** | $112 \times 112$ | $7 \times 7$ Convolution, 64 filters, stride 2 | Same
| **Pooling** | $56 \times 56$ | Same
| **Conv2_x** | $56 \times 56$ | `[3x3, 64; 3x3, 64] x 2`| Same |
| **Conv3_x** | $28 \times 28$ |`[3x3, 128; 3x3, 128] x 2` | Same | 
| **Conv4_x** | $14 \times 14$ |`[3x3, 256; 3x3, 256] x 2` | Same |
| **Conv5_x** | $7 \times 7$ |`[3x3, 512; 3x3, 512] x 2` | Same |
| **Output** | $1 \times 1$ | Average Pool $\rightarrow$ 1000-d FC $\rightarrow$ Softmax | Average Pool $\rightarrow$ 5-d FC; Softmax removed


### Trained version
- *biomass-baseline*
The model was trained on the CSIRO biomass prediction dataset.
This dataset contains 2000x1000 images. To train this model, we random cropped the images to 448x448 images and then resized them to 224x224 to get them to the standard ResNet18 input size.
To validate we centercropped the images to 448x448 and then resized to 224x224.
We also scaled the targets by 2000.

#### Results

|Statistic| biomass-baseline | 
| :--- | :--- |
| Val MSE | 0.027 |
| Train MSE | 0.005 | 
| Training Time | 32m | 