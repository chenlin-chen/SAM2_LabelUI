#!/bin/bash

python app.py --port 7890 --src_img_dir_name "SrcImgs" --checkpoint_dir "./checkpoints/sam2.1_hiera_large.pt" --model_cfg "configs/sam2.1/sam2.1_hiera_l.yaml"
