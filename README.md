## [Progressive Image Deraining Networks: A Better and Simpler Baseline]()

### Introduction
This paper provides a better and simpler baseline deraining network by discussing network architecture, input and output, and loss functions.
Specifically, by repeatedly unfolding a shallow ResNet, progressive ResNet (PRN) is proposed to take advantage of recursive computation.
A recurrent layer is further introduced to exploit the dependencies of deep features across stages, forming our progressive recurrent network (PReNet).
Furthermore, intra-stage recursive computation of ResNet can be adopted in PRN and PReNet to notably reduce network parameters with graceful degradation in deraining performance.
For network input and output, we take both stage-wise result and original rainy image as input to each ResNet and finally output the prediction of residual image.
As for loss functions, single MSE or negative SSIM losses are sufficient to train PRN and PReNet.
Experiments show that PRN and PReNet perform favorably on both synthetic and real rainy images.
Considering its simplicity, efficiency and effectiveness, our models are expected to serve as a suitable baseline in future deraining research. 


## Prerequisites
- Python 3.6, PyTorch >= 0.4.0 
- Requirements: opencv-python, tensorboardX
- Platforms: Ubuntu 16.04, cuda-8.0 & cuDNN v-5.1 (higher versions also work well)
- MATLAB for computing [evaluation metrics](statistic/)


## Datasets

PRN and PReNet are evaluated on four datasets*: Rain100H [1], Rain100L [1], Rain12 [2] and Rain1400 [3]. Please download the testing datasets from [BaiduYun](https://pan.baidu.com/s/1J0q6Mrno9aMCsaWZUtmbkg), and place the unzipped folders into `./test/`.

To train the models, please download training datasets: RainTrainH [1], RainTrainL [1] and Rain12600 [3] from [BaiduYun](https://pan.baidu.com/s/1J0q6Mrno9aMCsaWZUtmbkg), and place the unzipped folders into `./train/`. 

*_We note that:
(i) The datasets in the website of [1] seem to be modified. But the models and results in recent papers are all based on the previous version, and thus we upload the original training and testing datasets to [BaiduYun](https://pan.baidu.com/s/1J0q6Mrno9aMCsaWZUtmbkg). 
(ii) For RainTrainH, we strictly exclude 546 rainy images that have the same background contents with testing images.
All our models are trained on remaining 1,254 training samples._


## Getting Started

### 1) Testing

We have placed our pre-trained models into `./logs/`. 

Run shell scripts to test the models:
```bash
bash test_PRN.sh      # test PRN on four datasets
bash test_PReNet.sh   # test PReNet on four datasets
bash test_PRN_r.sh    # test PRN_r on four datasets
bash test_PReNet_r.sh # test PReNet_r on four datasets 
bash test_ablation.sh # test the models in Ablation Study
bash test_real.sh     # test PReNet on real rainy images
```
All the results in the paper are also available at [BaiduYun](https://pan.baidu.com/s/1Oym9G-8Bq-0FU2BfbARf8g).
You can place the downloaded results into `./results/`, and directly compute all the [evaluation metrics](statistic/) in this paper.  

### 2) Training

Run shell scripts to train the models:
```bash
bash train_PRN.sh      # train PRN on three datasets
bash train_PReNet.sh   # train PReNet on three datasets
bash train_PRN_r.sh    # train PRN_r on three datasets (may need several tries on Rain12600)
bash train_PReNet_r.sh # train PReNet_r on three datasets
bash train_ablation.sh # train the models in Ablation Study
```

### 3) Evaluation metrics

We also provide the MATLAB scripts to compute the average PSNR and SSIM values reported in the paper.
 

```Matlab
 cd ./statistic
 run statistic_Rain100H.m
 run statistic_Rain100L.m
 run statistic_Rain12.m
 run statistic_Rain1400.m
```
### Model Configuration

The following tables provide the configurations of options. 

#### Training Mode Configurations

Option                 |Default        | Description
-----------------------|---------------|------------
batchSize              | 16            | Training batch size
recurrent_iter         | 6             | Number of recursive stages
epochs                 | 100           | Number of training epochs
milestone              | [30,50,80]    | When to decay learning rate
lr                     | 1e-3          | Initial learning rate
save_freq              | 1             | save intermediate model
use_GPU                | True          | use GPU or not
gpu_id                 | 0             | GPU id
data_path              | N/A           | path to training images
save_path              | N/A           | path to save models and status           

#### Testing Mode Configurations

Option                 |Default           | Description
-----------------------|------------------|------------
use_GPU                | True             | use GPU or not
gpu_id                 | 0                | GPU id
recurrent_iter         | 6                | Number of recursive stages
data_path              | N/A              | path to testing images
save_path              | N/A              | path to save results

## References
[1] Yang W, Tan RT, Feng J, Liu J, Guo Z, Yan S. Deep joint rain detection and removal from a single image. In IEEE CVPR 2017.

[2] Li Y, Tan RT, Guo X, Lu J, Brown MS. Rain streak removal using layer priors. In IEEE CVPR 2016.

[3] Fu X, Huang J, Zeng D, Huang Y, Ding X, Paisley J. Removing rain from single images via a deep detail network. In IEEE CVPR 2017.

