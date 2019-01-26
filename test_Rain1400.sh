#! bash

# PReNet
python test_PReNet.py --logdir logs/Rain1400/PReNet6 --save_path results/Rain1400/PReNet --data_path datasets/test/Rain1400/rainy_image

# PReNet_r
python test_PReNet_r.py --logdir logs/Rain1400/PReNet6_r --save_path results/Rain1400/PReNet_r --data_path datasets/test/Rain1400/rainy_image

# PRN
python test_PRN.py --logdir logs/Rain1400/PRN6 --save_path results/Rain1400/PRN6 --data_path datasets/test/Rain1400/rainy_image

# PRN_r
python test_PRN_r.py --logdir logs/Rain1400/PRN6_r --save_path results/Rain1400/PRN_r --data_path datasets/test/Rain1400/rainy_image
