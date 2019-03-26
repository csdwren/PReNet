#! bash

# PReNet1
python test_real.py --logdir logs/real --which_model PReNet1.pth --save_path results/real1 --data_path datasets/test/real

# PReNet2
python test_real.py --logdir logs/real --which_model PReNet2.pth --save_path results/real2 --data_path datasets/test/real

