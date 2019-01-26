The datasets can be downloaded from https://pan.baidu.com/s/1J0q6Mrno9aMCsaWZUtmbkg or https://1drv.ms/f/s!AqLfQqtZ6GwGgep-hgjLxkov2SSZ3g. 

Please put the unzipped folders (RainTrainH, RainTrainL and Rain12600) into ./datasets/train/, and put the unzipped folders (Rain100H, Rain100L, Rain1400 and Rain12) into ./datasets/test/  



1. RainTrainH and Rain100H
RainTrainH and Rain100H are from http://www.icst.pku.edu.cn/struct/Projects/joint_rain_removal.html.
RainTrainH has 1800 rainy images for training, and Rain100H has 100 rainy images for testing. 

However, we find that 546 rainy images from the 1,800 training samples have the same background contents with testing images.
Therefore, we exclude these 546 images from training set, and train all our models on the remaining 1,254 training images.



2. RainTrainL and Rain100L 
RainTrainL and Rain100L are from http://www.icst.pku.edu.cn/struct/Projects/joint_rain_removal.html.
RainTrainL has 200 rainy images for training, and Rain100L has 100 rainy images for testing.

These datasets at http://www.icst.pku.edu.cn/struct/Projects/joint_rain_removal.html seem to be modified. But we still use the original one, since the models and results in recent papers are all based on the original dataset. 



3. Rain12600 and Rain1400
This dataset contains 1,000 clean images. Each clean image was used to generate 14 rainy images with different streak orientations and magnitudes.
Rain12600 has 900 clean images for "training" and Rain1400 has 100 clean images for "testing". 
This dataset is from https://xmu-smartdsp.github.io/



4. Rain12
This dataset only includes 12 rainy images from http://yu-li.github.io/paper/li_cvpr16_rain.zip


