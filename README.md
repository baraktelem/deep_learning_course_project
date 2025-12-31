# deep_learning_course_project
In our project we plan to examine the integration of wavelet kernels into CNN architectures. By integrating wavelets kernels to CNN layers, we aim to offload the texture detection task to the wavelet kernels, freeing the standard kernels to perform other tasks.

# Baselines
We trained 4 variations of two baseline models.
## CIFAR10 Adapted ResNet18 Model
### Adaptions
| Layer Name | Original Output Size | Original Configuration | Adapted Output Size | Adapted Configuration |
| :--- | :--- | :--- | :--- | :--- |
| **Conv1** | $112 \times 112$ | $7 \times 7$ Convolution, 64 filters, stride 2 | $32\times 32$ |$3\times3$ Convolution, 64 filters, stride 1 |
| **Pooling** | $56 \times 56$ | $3 \times 3$ Max Pooling, stride 2 | -| Removed |
| **Conv2_x** | $56 \times 56$ | `[3x3, 64; 3x3, 64] x 2`| $16 \times 16$ |Same |
| **Conv3_x** | $28 \times 28$ | `[3x3, 128; 3x3, 128] x 2`| $8 \times 8$ | Same | 
| **Conv4_x** | $14 \times 14$ |`[3x3, 256; 3x3, 256] x 2` | $4 \times 4$ | Same |
| **Conv5_x** | $7 \times 7$ |`[3x3, 512; 3x3, 512] x 2` | $2 \times 2$ | Same |
| **Output** | $1 \times 1$ | Average Pool $\rightarrow$ 1000-d FC $\rightarrow$ Softmax |$1 \times 1$ | Average Pool $\rightarrow$ 10-d FC $\rightarrow$ Softmax |

**Total Parameters:** 11,173,962
**Model Size:** 42.63 MB

### Trained versions
All versions were trained on the CIFAR10 train dataset with basic augmentation for 200 epochs and evaluated on the CIFAR10 test dataset.

- *baseline.ipynb*: Trained on 4000 samples of each class.
- *baseline-500.ipynb*: Trained on 500 samples of each class.
- *baseline-100.ipynb*: Trained on 100 samples of each class.

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
| **Pooling** | $56 \times 56$ | $3 \times 3$ Max Pooling, stride 2 | Same |
| **Conv2_x** | $56 \times 56$ | `[3x3, 64; 3x3, 64] x 2`| Same |
| **Conv3_x** | $28 \times 28$ |`[3x3, 128; 3x3, 128] x 2` | Same | 
| **Conv4_x** | $14 \times 14$ |`[3x3, 256; 3x3, 256] x 2` | Same |
| **Conv5_x** | $7 \times 7$ |`[3x3, 512; 3x3, 512] x 2` | Same |
| **Output** | $1 \times 1$ | Average Pool $\rightarrow$ 1000-d FC $\rightarrow$ Softmax | Average Pool $\rightarrow$ 5-d FC; Softmax removed |


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

# ScatResNet18 v1
## CIFAR10 Adapted ScatResNet18 v1
We started with a small modification of the first convolutional layer in the CIFAR10 Adapted ResNet18 model:
We swapped some standard convolutional channels in the first convolutional layer with scattering channels. 
Because the scattering transform also downsamples its input, we upsample it back using a deconvolutional layer to connect it back to the model. (Other possible options are non-parameteric upsampling, and reshaping) 
The model has a hyperparameter $L$ that determines the number of channels swapped by determining the number of kernel rotations in the scattering transform. (Other possible options are adding channels with outputs from larger wavelet kernels)

## Adaptions
| Layer Name | ResNet18 | ScatResNet18 v1 | ScatResNet18 v1 
| :--- | :--- | :--- | :--- | 
| **Conv1** | $3\times3$ Convolution, 64 filters, stride 1 | $ 3 \times 3$ Convolution, $\left(64 - (L+1)\cdot 3\right)$ filters. | $(L+1)\cdot 3$ Channels Scatternet with output dimension $16\times 16$|
| **Deconv1** | New to ScatResNet | - | $2 \times 2$ Deconv with stride $2$ and number of groups . Output size $32\times 32$ |
| **Conv2_x** |  `[3x3, 64; 3x3, 64] x 2`| Deconv and Conv outputs are concatinated and inputted to this same block |
| **Conv3_x** |  `[3x3, 128; 3x3, 128] x 2`| Same | 
| **Conv4_x** | `[3x3, 256; 3x3, 256] x 2` | Same |
| **Conv5_x** | `[3x3, 512; 3x3, 512] x 2` | Same |
| **Output** |  Average Pool $\rightarrow$ 10-d FC $\rightarrow$ Softmax | Same |


## Results
### Training on 100 samples for each class `ScatResNet18_v1_100.ipynb`
| L | Test Acc |
|:---|:---|
| 1 | 49.670 |
| 2 | 54.760 |
| 3 | 54.760 |
| 4 | 61.840 |
| 5 | 55.920 |
| 6 | 55.280 |
| 7 | 59.210 |
| 8 | 57.640 |
| 9 | 55.550 |
| 10 | 61.690 |
| 11 | 55.840 |
| 12 | 58.270 |
| 13 | 57.800 |
| 14 | 56.550 |
| 15 | 55.680 |
| 16 | 57.170 |
| 17 | 56.600 |
| 18 | 55.790 |
| 19 | 58.220 |
| 20 | 57.430 |

Looks like the best L value is 4 with Test Acc 61.84%, which is a significant improvement over the respective baseline that achieved only 46.10% Test Acc. This is expected since scattering networks are known to work better than CNNs in lower data regimes. \
The results look somewhat random, and we suspect training results to be very sensitive to weights initializations. To test that we created the notebook `ScatResNet18_v1_100_is_random.ipynb`, and trained the model 6 times for L values of [4, 6, 8, 10]. Those are the resulting mean and variance:
| L | Mean | Variance |
|:---| :---| :---|
|4| 55.65 | 0.426|
|6| 56.09 | 4.755|
|8| 58.17 | 0.971|
|10| 57.94 | 2.133|
Meaning the best value L we found in the first run has the worst mean, and some values have high variance. The results show fairly well that initialization plays a significant role in this architecture. 

### Training on 500 samples for each class 
`ScatResNet18_v1_500.ipynb`

### Training on full CIFAR10 
`ScatResNet18_v1.ipynb`

|Model| Test Acc | Validation Acc | Train Acc |
|:---|:---|:---|:---|
Baseline| 94.14 | 94.89 | Fill |
ScatResNet18 L=4 | 93.46 | 94.19 | 100 |
ScatResNet18 L=10 | 93.97 | 94.55 | 100 |

## Results Analysis
When proposing this architecture, we expected it to perform better on lower data regimes and worse on larger data regims. 
This is because, with enough samples, the CNN can learn filters which are more adapted to the dataset, that perform better than the general wavelet kernels. On the other hand, without enough samples, the predefined wavelet kernels does not require learning and provide better features than those that can be learned by the CNN kernels.
The results align with our predictions.
In a larger data regime, this model does not improve on the baseline, and performs slightly worse than it.
And in the lower data regimes, this model significantly improves the baselines.