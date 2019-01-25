#! bash

# Rain100H
python train_PReNet_r.py --save_path logs/Rain100H/PReNet_r --data_path datasets/train/RainTrainH

# Rain100L
python train_PReNet_r.py --save_path logs/Rain100L/PReNet_r --data_path datasets/train/RainTrainL

# Rain12600
python train_PReNet_r.py --save_path logs/Rain1400/PReNet_r --data_path datasets/train/Rain12600
