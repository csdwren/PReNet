## [Progressive Image Deraining Networks: A Better and Simpler Baseline](https://www.researchgate.net/publication/338511165_Progressive_Image_Deraining_Networks_A_Better_and_Simpler_Baseline) 
[[arxiv](https://arxiv.org/abs/1901.09221)] [[pdf](https://www.researchgate.net/publication/338511165_Progressive_Image_Deraining_Networks_A_Better_and_Simpler_Baseline)] [[supp](https://csdwren.github.io/papers/PReNet_supp.pdf)]

### Introduction
This paper provides a better and simpler baseline deraining network by discussing network architecture, input and output, and loss functions.
Specifically, by repeatedly unfolding a shallow ResNet, progressive ResNet (**PRN**) is proposed to take advantage of recursive computation.
A recurrent layer is further introduced to exploit the dependencies of deep features across stages, forming our progressive recurrent network (**PReNet**).
Furthermore, intra-stage recursive computation of ResNet can be adopted in PRN and PReNet to notably reduce network parameters with graceful degradation in deraining performance (**PRN_r** and **PReNet_r**).
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

PRN and PReNet are evaluated on four datasets*: 
Rain100H [1], Rain100L [1], Rain12 [2] and Rain1400 [3]. 
Please download the testing datasets from [BaiduYun](https://pan.baidu.com/s/1J0q6Mrno9aMCsaWZUtmbkg)
or [OneDrive](https://1drv.ms/f/s!AqLfQqtZ6GwGgep-hgjLxkov2SSZ3g), 
and place the unzipped folders into `./datasets/test/`.

To train the models, please download training datasets: 
RainTrainH [1], RainTrainL [1] and Rain12600 [3] from [BaiduYun](https://pan.baidu.com/s/1J0q6Mrno9aMCsaWZUtmbkg)
or [OneDrive](https://1drv.ms/f/s!AqLfQqtZ6GwGgep-hgjLxkov2SSZ3g), 
and place the unzipped folders into `./datasets/train/`. 

*_We note that:_

_(i) The datasets in the website of [1] seem to be modified. 
    But the models and results in recent papers are all based on the previous version, 
    and thus we upload the original training and testing datasets 
    to [BaiduYun](https://pan.baidu.com/s/1J0q6Mrno9aMCsaWZUtmbkg) 
    and [OneDrive](https://1drv.ms/f/s!AqLfQqtZ6GwGgep-hgjLxkov2SSZ3g)._ 

_(ii) For RainTrainH, we strictly exclude 546 rainy images that have the same background contents with testing images.
    All our models are trained on remaining 1,254 training samples._


## Getting Started

### 1) Testing

We have placed our pre-trained models into `./logs/`. 

Run shell scripts to test the models:
```bash
bash test_Rain100H.sh   # test models on Rain100H
bash test_Rain100L.sh   # test models on Rain100L
bash test_Rain12.sh     # test models on Rain12
bash test_Rain1400.sh   # test models on Rain1400 
bash test_Ablation.sh   # test models in Ablation Study
bash test_real.sh       # test PReNet on real rainy images
```
All the results in the paper are also available at [BaiduYun](https://pan.baidu.com/s/1Oym9G-8Bq-0FU2BfbARf8g).
You can place the downloaded results into `./results/`, and directly compute all the [evaluation metrics](statistic/) in this paper.  

### 2) Evaluation metrics

We also provide the MATLAB scripts to compute the average PSNR and SSIM values reported in the paper.
 

```Matlab
 cd ./statistic
 run statistic_Rain100H.m
 run statistic_Rain100L.m
 run statistic_Rain12.m
 run statistic_Rain1400.m
 run statistic_Ablation.m  # compute the metrics in Ablation Study
```
###
Average PSNR/SSIM values on four datasets:

Dataset    | PRN       |PReNet     |PRN_r      |PReNet_r   |JORDER[1]  |RESCAN[4]
-----------|-----------|-----------|-----------|-----------|-----------|-----------
Rain100H   |28.07/0.884|29.46/0.899|27.43/0.874|28.98/0.892|26.54/0.835|28.88/0.866
Rain100L   |36.99/0.977|37.48/0.979|36.11/0.973|37.10/0.977|36.61/0.974|---
Rain12     |36.62/0.952|36.66/0.961|36.16/0.961|36.69/0.962|33.92/0.953|---
Rain1400   |31.69/0.941|32.60/0.946|31.31/0.937|32.44/0.944| ---       |---

*_We note that:_

_(i) The metrics by JORDER[1] are computed directly based on the deraining images 
provided by the authors._ 

_(ii) RESCAN[4] is re-trained with their default settings: 
(1) RESCAN for Rain100H is trained on the full 1800 rainy images, while our models are all trained on the strict 1254 rainy images.
(2) The re-trained model of RESCAN is available at [here](https://pan.baidu.com/s/1Oym9G-8Bq-0FU2BfbARf8g)._
 
_(iii) The deraining results by JORDER and RESCAN can be downloaded 
from [here](https://pan.baidu.com/s/1Oym9G-8Bq-0FU2BfbARf8g), 
and their metrics in the above table can be computed by the [Matlab scripts](statistic/statistic_rain100H.m)._ 

### 3) Training

Run shell scripts to train the models:
```bash
bash train_PReNet.sh      
bash train_PRN.sh   
bash train_PReNet_r.sh    
bash train_PRN_r.sh  
```
You can use `tensorboard --logdir ./logs/your_model_path` to check the training procedures. 

### Model Configuration

The following tables provide the configurations of options. 

#### Training Mode Configurations

Option                 |Default        | Description
-----------------------|---------------|------------
batchSize              | 18            | Training batch size
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
logdir                 | N/A              | path to trained model
data_path              | N/A              | path to testing images
save_path              | N/A              | path to save results

## References
[1] Yang W, Tan RT, Feng J, Liu J, Guo Z, Yan S. Deep joint rain detection and removal from a single image. In IEEE CVPR 2017.

[2] Li Y, Tan RT, Guo X, Lu J, Brown MS. Rain streak removal using layer priors. In IEEE CVPR 2016.

[3] Fu X, Huang J, Zeng D, Huang Y, Ding X, Paisley J. Removing rain from single images via a deep detail network. In IEEE CVPR 2017.

[4] Li X, Wu J, Lin Z, Liu H, Zha H. Recurrent squeeze-and-excitation context aggregation net for single image deraining.In ECCV 2018.


# Citation

```
 @inproceedings{ren2019progressive,
   title={Progressive Image Deraining Networks: A Better and Simpler Baseline},
   author={Ren, Dongwei and Zuo, Wangmeng and Hu, Qinghua and Zhu, Pengfei and Meng, Deyu},
   booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
   year={2019},
 }
 ```
