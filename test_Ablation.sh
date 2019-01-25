#! bash 

# PReNet
python test_PReNet.py --logdir logs/Rain100H/PReNet6 --save_path results/Ablation/PReNet --data_path datasets/test/Rain100H/rainy

# PReNet6-RecSSIM
python test_PReNet_LSTM.py --logdir logs/Ablation/PReNet6_RecSSIM --save_path results/Ablation/PReNet_RecSSIM --data_path datasets/test/Rain100H/rainy

# PReNet6-LSTM
python test_PReNet_LSTM.py --logdir logs/Ablation/PReNet6_LSTM --save_path results/Ablation/PReNet6_LSTM --data_path datasets/test/Rain100H/rainy

# PReNet5-LSTM
python test_PReNet_LSTM.py --recurrent_iter 5 --logdir logs/Ablation/PReNet5_LSTM --save_path results/Ablation/PReNet5_LSTM --data_path datasets/test/Rain100H/rainy

# PReNet7-LSTM
python test_PReNet_LSTM.py --recurrent_iter 7 --logdir logs/Ablation/PReNet7_LSTM --save_path results/Ablation/PReNet7_LSTM --data_path datasets/test/Rain100H/rainy

# PReNet6-GRU
python test_PReNet_GRU.py --logdir logs/Ablation/PReNet6_GRU --save_path results/Ablation/PReNet6_GRU --data_path datasets/test/Rain100H/rainy

# PReNet6_x
python test_PReNet_x.py --logdir logs/Ablation/PReNet6_x --save_path results/Ablation/PReNet6_x --data_path datasets/test/Rain100H/rainy
