#! bash 

# PReNet
python test_PReNet.py --logdir logs/Rain100H/PReNet6 --save_path results/Rain100H/PReNet --data_path datasets/test/Rain100H/rainy

# PReNet_r
python test_PReNet_r.py --logdir logs/Rain100H/PReNet6_r --save_path results/Rain100H/PReNet_r --data_path datasets/test/Rain100H/rainy

# PRN
python test_PRN.py --logdir logs/Rain100H/PRN6 --save_path results/Rain100H/PRN6 --data_path datasets/test/Rain100H/rainy

# PRN_r
python test_PRN_r.py --logdir logs/Rain100H/PRN6_r --save_path results/Rain100H/PRN_r --data_path datasets/test/Rain100H/rainy
