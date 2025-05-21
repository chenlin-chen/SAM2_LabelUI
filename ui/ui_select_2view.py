import torch
# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# torch.multiprocessing.set_start_method("spawn")

import colorsys
import datetime
import os
import subprocess

import cv2
import json
import gradio as gr
import imageio.v2 as iio
import numpy as np
import shutil


from typing import Tuple
from enum import Enum
from loguru import logger as guru


from .ui_base import UI


def listdir(vid_dir):
    if vid_dir is not None and os.path.isdir(vid_dir):
        folders = [folder for folder in os.listdir(vid_dir) if os.path.isdir(os.path.join(vid_dir, folder))]
        return sorted(folders)

    return []


class ImageSelect2ViewUI(UI):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


        data_dir = kwargs.pop("data_dir", None)
        ref_img_dir = kwargs.pop("ref_img_dir", None)
        src_img_dir_name = kwargs.pop("src_img_dir_name", None)

        self.data_dir = data_dir
        self.ref_img_dir = ref_img_dir
        self.src_img_dir_name = src_img_dir_name
        self.src_img_dir = os.path.join(self.data_dir, src_img_dir_name)
        self.last_indice_filename  = "last_indices.json"
        self.record_select_filename = "select_result.json"

        self.file_order_filename = "sorted_by_skin_iou.txt"

        self.record_select_dict = {}

        self.img_dir = None
        self.img_paths = []
        self.img_ref_paths = []
        self.image = None
        self.display_image = None
        self.image_max_size = 1920
        self.img_index_current = -1


    def main(self,):
        # composition
        with gr.Blocks() as demo:
            img_dirs = listdir(self.src_img_dir)
            with gr.Row():
                with gr.Column():
                    img_dirs_dropdown = gr.Dropdown(
                        label="Image directories", choices=[], value="",
                    )

                    selected_img_name_text = gr.Text(
                        None, label="img_name", interactive=False,
                    )
                selected_img_dir_text = gr.Text(
                    None, label="Input directory", interactive=False, visible=False,
                )

                with gr.Column():
                    image_index_slider = gr.Slider(
                            label="Image index",
                            minimum=0,
                            maximum=0,
                            value=self.img_index_current,
                            step=1,
                        )
                    with gr.Row():
                        prev_btn = gr.Button("⏮️ Previous Image")
                        next_btn = gr.Button("Next Image ⏭️")


            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        display_image = gr.Image(label="Current selection", height=600,
                                                show_download_button=False)
                        display_image_ref = gr.Image(label="Reference", height=600,
                                            show_download_button=False)

                    with gr.Row():
                        correct_btn = gr.Button("Correct")
                        worng_btn = gr.Button("Wrong")


            # event
            '''
            selected_img_dir_text.change(
                self.update_image_dir,
                [selected_img_dir_text ],
                [image_index_slider, display_image, display_image_ref],
            )
            '''

            # selecting an image directory
            img_dirs_dropdown.select(
                self.select_image_dir,
                [img_dirs_dropdown ],
                [image_index_slider, display_image, display_image_ref],
            )

            # Button actions
            prev_btn.click(self.update_img_index_by_btn, inputs=[image_index_slider, gr.State("prev")], outputs=image_index_slider)
            next_btn.click(self.update_img_index_by_btn, inputs=[image_index_slider, gr.State("next")], outputs=image_index_slider)

            correct_btn.click(self.update_img_result_by_btn, inputs=[image_index_slider, gr.State("correct")], outputs=image_index_slider)
            worng_btn.click(self.update_img_result_by_btn, inputs=[image_index_slider, gr.State("worng")], outputs=image_index_slider)
            image_index_slider.change(self.update_img_index_event, [image_index_slider], [display_image, display_image_ref, selected_img_name_text])

            demo.load(self.update_img_dir_dropdown, outputs=[img_dirs_dropdown])

        return demo

    def resize_img(self, image, is_mask=False):
        h, w = image.shape[:2]
        max_size = self.image_max_size
        if max(h, w) > max_size:
            # Resize the image to make sure the max dimension is 1920
            if h > w:
                new_h = max_size
                new_w = int(w * (max_size / h))
            else:
                new_w = max_size
                new_h = int(h * (max_size / w))

            if is_mask:
                image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            else:
                image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            image = np.copy(image)

        return image
    
    def update_show_img(self):
        def set_input_image(i: int = 0):
            guru.debug(f"Setting image_index {i} / {len(self.img_paths)}")
            if i < 0 or i >= len(self.img_paths):
                return

            image = iio.imread(self.img_paths[i])
            if image.shape[-1] == 4:
                image = image[:, :, :3]

            self.image = image
            self.display_image = self.resize_img(image)

            image_ref = None
            if len(self.img_ref_paths) > 0:
                image_ref = iio.imread(self.img_ref_paths[i])

            return image_ref

        image_ref = set_input_image(self.img_index_current)
        return self.display_image, image_ref

    def update_img_index_event(self, img_index):
        self.img_index_current = img_index        

        if self.img_index_current > -1:
            display_image, display_image_label = self.update_show_img()
            img_name = os.path.basename(self.img_paths[img_index])
        else:
            display_image = None  # Handle empty directory case
            display_image_label = None
            img_name = None

        self.save_last_indice()
            
        return display_image, display_image_label, img_name

        
    def update_img_index_by_btn(self, current_value, direction):
        if direction == "prev":
            return min(max(current_value - 1, 0), len(self.img_paths)-1)  # Prevent going below 0
        elif direction == "next":
            return min(current_value + 1, len(self.img_paths)-1)

    def update_img_result_by_btn(self, current_value, direction):
        filename = os.path.basename(self.img_paths[current_value])
        if direction == "correct":
            self.record_select_dict[filename]=True

        elif direction == "worng":
            self.record_select_dict[filename]=False

        self.save_record_selected()
        return min(max(current_value + 1, 0), len(self.img_paths) - 1)  # Prevent going below 0

        
    def select_image_dir(self, img_dir_name):
        select_img_dir = os.path.join(self.src_img_dir, img_dir_name)
        guru.debug(f"Selected image dir: {select_img_dir}")

        json_file_path = os.path.join(select_img_dir, self.record_select_filename)
        if os.path.exists(json_file_path):
            with open(json_file_path, 'r') as file:
                self.record_select_dict = json.load(file)
        else:
            self.record_select_dict = {}


        #return select_img_dir
        return self.update_image_dir(select_img_dir)

    def set_img_dir(self, img_dir: str) -> int:
        def isimage(p):
            ext = os.path.splitext(p.lower())[-1]
            return ext in [".png", ".jpg", ".jpeg"]

        self.image = None
        self.display_image = None
        self.img_dir = img_dir
        self.img_paths = [
            f"{img_dir}/{p}" for p in sorted(os.listdir(img_dir)) if isimage(p)
        ]


        #img_ref_dir = img_dir + "_label"
        img_ref_dir = os.path.join(self.ref_img_dir, os.path.basename(img_dir))
        file_order_path = os.path.join(img_dir, self.file_order_filename)
        if os.path.exists(file_order_path):
            with open(file_order_path, 'r') as f:
                ordered_paths = []
                ordered_ref_paths = []                
                for line in f :
                    img_name = line.strip().split(": ")[0]
                    ordered_paths.append(os.path.join(img_dir, img_name))
                    ordered_ref_paths.append(os.path.join(img_ref_dir, img_name))

                #ordered_paths  = [ 
                #    os.path.join(img_dir, line.strip().split(": ")[0]) for line in f 
                #]
                self.img_paths = ordered_paths
                self.img_ref_paths = ordered_ref_paths

        else:
            # 如果排序文件不存在，根据已有图像路径生成参考图像路径
            ref_paths = []
            for img_path in self.img_paths:
                img_name = os.path.basename(img_path)
                name_without_ext, _ = os.path.splitext(img_name)
                img_name_jpg = name_without_ext + ".jpg"
                img_name_png = name_without_ext + ".png"
                ref_img_path = os.path.join(img_ref_dir, img_name_jpg)
                ref_img_png_path = os.path.join(img_ref_dir, img_name_png)

                if os.path.exists(ref_img_path):
                    ref_paths.append(ref_img_path)
                elif os.path.join(ref_img_png_path):
                    ref_paths.append(ref_img_png_path)
                else:
                    raise FileNotFoundError(f"Reference image not found: {name_without_ext} in fodler {img_ref_dir}")
                    
            self.img_ref_paths = ref_paths


        return len(self.img_paths)

    def update_image_dir(self, img_dir):
        print(f"update_image_dir img_dir {img_dir}")
        self.set_img_dir(img_dir)
        img_index_old = self.img_index_current
        #self.img_index_current = self.get_latest_processed_img_index(img_dir)
        self.img_index_current = self.get_last_img_index(img_dir)

        # enable_force_update ensures the front-end updates if the image index is valid and unchanged,
        # otherwise gr.update triggers a refresh with update_show_img.
        enable_force_update = self.img_index_current > -1 and img_index_old == self.img_index_current
        if enable_force_update:
            display_image, display_image_ref = self.update_show_img()
        else:
            display_image = None  # Handle empty directory case
            display_image_ref = None

        return gr.update(maximum=len(self.img_paths)- 1, value=self.img_index_current), display_image, display_image_ref


    def save_last_indice(self):
        if self.img_dir is not None:
            last_indices_file = os.path.join(self.img_dir, self.last_indice_filename)
            #guru.debug(f"save last_indice_filename: {last_indices_file}")
            with open(last_indices_file, 'w') as f:
                json.dump(self.img_index_current, f)

    def save_record_selected(self):
        if self.img_dir is not None:
            record_select_file = os.path.join(self.img_dir, self.record_select_filename)
            with open(record_select_file, 'w') as f:
                json.dump(self.record_select_dict, f)

    def get_last_img_index(self, img_dir):

        last_indices_file = os.path.join(img_dir, self.last_indice_filename)
        
        if os.path.exists(last_indices_file):
            with open(last_indices_file, 'r') as f:
                last_img_index = json.load(f)
        else:
            last_img_index = 0  # Default to 0 if no file exists

        return last_img_index    


    def update_img_dir_dropdown(self):
        img_dirs = listdir(self.src_img_dir)

        return  gr.update(choices=img_dirs)

