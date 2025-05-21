#!/bin/bash

# enable testbed label mask
python app.py --port 7890 --data_dir /data/SkinParsing --src_img_dir_name "SrcImgs/" --ui_mode ui_seg 

# enable testbed select
#python app.py --port 7890 --data_dir /notebooks/SkinParsing/facer --src_img_dir_name "CheckFaceParsing" --ui_mode ui_select 


# enable testbed select_2view
#python app.py --port 7890 --data_dir /notebooks/SkinParsing/FaceSegResult \
#--src_img_dir_name "ffhq_test" \
#--ref_img_dir /data/FaceSkinParsing/ffhq/images1024x1024 \
#--ui_mode ui_select_2view 