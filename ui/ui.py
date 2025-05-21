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

from typing import Tuple
from enum import Enum
from loguru import logger as guru

try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    #from segment_anything import SamPredictor, sam_model_registry
except ImportError as e:
    print(f"Error importing module: {e}")
    
from .ui_base import UI


class PromptGUI(object):
    def __init__(self, sam_predictor, label_class, mask_th=0.0):
        self.sam_predictor = sam_predictor

        self.selected_points = []
        self.selected_labels = []
        self.cur_label_val = 1.0
        self.mask_th = mask_th
        self.label_class = label_class

        self.cur_mask_idx = 0
        self.previous_mask = None
        self.index_mask = None

        # can store multiple object masks
        # saves the masks and logits for each mask index
        self.cur_masks = []
        self.cur_logits = []

    def clear_points(self) -> tuple[None, None, str]:
        self.selected_points.clear()
        self.selected_labels.clear()

        self.previous_mask = None
        self.cur_masks = []
        self.cur_logits = []
        self.index_mask = None
        message = "Cleared points, select new points to update mask"

        return message

    def add_new_mask(self):
        self.cur_mask_idx += 1
        self.clear_points()
        message = f"Creating new mask with index {self.cur_mask_idx}"
        return None, message

    def make_index_mask(self, masks):
        assert len(masks) > 0
        idcs = list(masks.keys())
        idx_mask = masks[idcs[0]].astype("uint8")
        for i in idcs:
            mask = masks[i]
            idx_mask[mask] = i + 1
        return idx_mask

    def clear_image(self):
        """
        clears image and all masks/logits for that image
        """
        self.index_mask = None
        self.previous_mask = None
        self.cur_mask_idx = 0   
        self.cur_masks = []
        self.cur_logits = []

    def reset(self):
        self.clear_image()
        self.clear_points()


    def set_positive(self):
        self.cur_label_val = 1.0

    def set_negative(self):
        self.cur_label_val = 0.0

    def add_point(self, i, j):
        """
        get the index mask of the objects
        """
        self.selected_points.append([j, i])
        self.selected_labels.append(self.cur_label_val)

        # masks, scores, logits if we want to update the mask
        masks, low_res_mask = self.get_sam_mask(
            self.previous_mask, np.array(self.selected_points, dtype=np.float32), np.array(self.selected_labels, dtype=np.int32)
        )

        self.index_mask = self.make_index_mask(masks)
        self.previous_mask = low_res_mask

        return self.index_mask

    def undo_points(self):  
        if len(self.selected_points) > 1:
            self.selected_points.pop()
            self.selected_labels.pop()

            # masks, scores, logits if we want to update the mask
            masks, low_res_mask = self.get_sam_mask(
                self.previous_mask, np.array(self.selected_points, dtype=np.float32), np.array(self.selected_labels, dtype=np.int32)
            )

            self.index_mask = self.make_index_mask(masks)
            self.previous_mask = low_res_mask
        else:
            self.clear_points()
            self.index_mask = None

        return self.index_mask
    

    def get_sam_mask(self, previous_mask, input_points, input_labels):
        """
        :param frame_idx int
        :param input_points (np array) (N, 2)
        :param input_labels (np array) (N,)
        return (H, W) mask, (H, W) logits
        """
        assert self.sam_predictor is not None
        

        if torch.cuda.get_device_properties(0).major >= 8:
            # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, scores, low_res_mask  = self.sam_predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                mask_input=previous_mask,
                multimask_output=False,
                return_logits=True,
            )

            logits = logits[[np.argmax(scores)]]
            low_res_mask = low_res_mask[[np.argmax(scores)]]

            masks = logits > self.mask_th
            self.cur_masks = [(mask, f'_{self.label_class}_mask_{i}') for i, mask in enumerate(masks)]
            #self.cur_logits = [(logit, f'_{self.label_class}_logit_{i}') for i, logit in enumerate(logits)]

        return  { self.cur_mask_idx : masks.squeeze()}, low_res_mask


    def save_mask_to_dir(self, output_dir: str, src_img_path: str) -> str:
        assert self.cur_masks is not None

        save_dir = os.path.join(output_dir, self.label_class)
        os.makedirs(save_dir, exist_ok=True)
        img_name = os.path.splitext(os.path.basename(src_img_path))[0]

        for mask, ext_name in self.cur_masks:
            o_mask = (mask * 255).astype(np.uint8)
            out_path = os.path.join(save_dir, img_name) + '.png'
            iio.imwrite(out_path, o_mask)

        '''
        for logit, ext_name in self.cur_logits:
            o_logit = (logit * 255).clip(0, 255).astype(np.uint8)
            out_path = os.path.join(save_dir, img_name+"_logit") + '.png'
            iio.imwrite(out_path, o_logit)
        '''

        message = f"Saved masks to {save_dir}!"
        if len(self.cur_masks) > 0:        
            guru.debug(message)
        return message



def draw_points(img, points, labels, circle_r=10):
    out = img.copy()
    for p, label in zip(points, labels):
        x, y = int(p[0]), int(p[1])
        color = (0, 255, 0) if label == 1.0 else (255, 0, 0)
        out = cv2.circle(out, (x, y), circle_r, color, -1)
    return out


def get_hls_palette(
    n_colors: int,
    lightness: float = 0.5,
    saturation: float = 0.7,
) -> np.ndarray:
    """
    returns (n_colors, 3) tensor of colors,
        first is black and the rest are evenly spaced in HLS space
    """
    hues = np.linspace(0, 1, int(n_colors) + 1)[1:-1]  # (n_colors - 1)
    # hues = (hues + first_hue) % 1
    palette = [(0.0, 0.0, 0.0)] + [
        colorsys.hls_to_rgb(h_i, lightness, saturation) for h_i in hues
    ]
    return (255 * np.asarray(palette)).astype("uint8")


def colorize_masks(images, index_masks, fac: float = 0.5):
    max_idx = max([m.max() for m in index_masks])
    guru.debug(f"{max_idx=}")
    palette = get_hls_palette(max_idx + 1)
    color_masks = []
    out_frames = []
    for img, mask in zip(images, index_masks):
        clr_mask = palette[mask.astype("int")]
        color_masks.append(clr_mask)
        out_u = compose_img_mask(img, clr_mask, fac)
        out_frames.append(out_u)
    return out_frames, color_masks


def compose_img_mask(img, color_mask, fac: float = 0.5):
    out_f = fac * img / 255 + (1 - fac) * color_mask / 255
    out_u = (255 * out_f).astype("uint8")
    return out_u

def listdir(vid_dir):
    if vid_dir is not None and os.path.isdir(vid_dir):
        folders = [folder for folder in os.listdir(vid_dir) if os.path.isdir(os.path.join(vid_dir, folder))]
        return sorted(folders)
    return []

class CustomRadio(gr.Radio):
    def __init__(self, choices=None, **kwargs):
        # Initialize the parent class (gr.Radio) with the given choices and other parameters
        super().__init__(choices=choices, **kwargs)
    
    # Override the update method to add custom behavior (if needed)
    def update(self, *args, **kwargs):
        print(f"Updating CustomRadio with args: {args}, kwargs: {kwargs}")
        super().update(*args, **kwargs)  # Call the parent class update method
    
    # Example of adding a custom method
    def custom_method(self):
        print("This is a custom method in the CustomRadio class.")


class LabelMode(Enum):
    #PERSON = "Person"
    FULL_BODY_SKIN = "FullBodySkin"
    #HAND_PART_SKIN = "HandPartSkin"  # Added new label for "手部"
    SKIN_UNDER_LENSES  = "SkinUnderLenses"
    BIG_BEARD = "BigBeard"
    FACE_TATOO = "FaceTattoo"

class ImageSegUI(UI):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


        checkpoint_dir = kwargs.pop("checkpoint_dir", None)
        model_cfg = kwargs.pop("model_cfg", None)
        data_dir = kwargs.pop("data_dir", None)
        # Raise an error if mandatory arguments are missing
        if checkpoint_dir is None or model_cfg is None or data_dir is None:
            raise ValueError("checkpoint_dir, model_cfg and data_dir are required arguments")

        src_img_dir_name = kwargs.pop("src_img_dir_name", None)
        output_dir_name = kwargs.pop("output_dir_name", None)
        if checkpoint_dir is None or output_dir_name is None:
             raise ValueError("src_img_dir_name and output_dir_name are required arguments")

        self.mask_th = 0.0        
        self.data_dir = data_dir
        self.src_img_dir_name = src_img_dir_name        
        self.src_img_dir = os.path.join(self.data_dir, src_img_dir_name)

        self.dst_dir_root = os.path.join(self.data_dir, output_dir_name)
        self.dst_mask_dir_name = "masks"
        self.dst_mask_dir = os.path.join(self.dst_dir_root, self.dst_mask_dir_name)

        self.img_dir = None
        self.img_paths = []
        self.image = None
        self.display_image = None
        self.image_max_size = 1920
        self.img_index_current = -1

        self.last_indice_filename  = "last_indices.json"

        self.processed_indices = set()  # Track processed image indices
        self.processed_indices_filename  = "processed_indices.json"


        self.sam_predictor = self.init_sam_model(model_cfg, checkpoint_dir)
        self.prompts_label = self.SetPromptLabels()

    def init_sam_model(self, model_cfg, checkpoint_dir):
        sam_predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint_dir))
        #guru.info(f"loaded model checkpoint {self.checkpoint_dir}")

        #sam_predictor = SamPredictor(sam_model_registry["vit_h"](checkpoint=self.checkpoint_dir))
        return sam_predictor

    def SetPromptLabels(self):
        # Create a dictionary to map each LabelMode to its respective PromptGUI instance

        prompts_label = {mode.value: PromptGUI(self.sam_predictor, mode.value, self.mask_th) for mode in LabelMode}
        return prompts_label

    def main(self,):
        with gr.Blocks() as demo:
            with gr.Column():
                with gr.Row():
                    data_dir_field = gr.Label(self.data_dir, label="Dataset root directory", elem_id="custom-label")
                    src_dir_field = gr.Label(self.src_img_dir_name, label="Image subdirectory name", elem_id="custom-label")
                    mask_dir_field = gr.Label(self.dst_mask_dir_name, label="Mask subdirectory name", elem_id="custom-label")

            
            with gr.Row():
                img_dirs_dropdown = gr.Dropdown(
                    label="Image directories", choices=[], value=None,
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


            with gr.Tabs():
                with gr.Tab("1st Step - SAM"):
                    sam_display, instruction = self.ui_SAM(mask_dir_field, image_index_slider)

            selected_img_dir_text.change(
                self.update_image_dir,
                [selected_img_dir_text ],
                [image_index_slider, sam_display],
            )

            # selecting an image directory
            img_dirs_dropdown.select(
                self.select_image_dir,
                [img_dirs_dropdown ],
                [selected_img_dir_text , mask_dir_field],
            )

            # Button actions
            prev_btn.click(self.update_img_index_by_btn, inputs=[image_index_slider, gr.State("prev")], outputs=image_index_slider)
            next_btn.click(self.update_img_index_by_btn, inputs=[image_index_slider, gr.State("next")], outputs=image_index_slider)
            image_index_slider.change(self.update_img_index_event, [image_index_slider], [sam_display, instruction])

            demo.load(self.update_img_dir_dropdown, outputs=[img_dirs_dropdown])

        return demo


    def ui_SAM(self, mask_dir_field, image_index_slider):        
        with gr.Row():
            with gr.Column():
                display_view = gr.Image(label="Current selection", height=700, 
                    show_download_button=False)

            with gr.Column():
                label_mode_radio = gr.Radio(
                    choices=[mode.value for mode in LabelMode],  # List of enum values
                    label="Select Label Mode",  # Label for the radio buttons
                    type="value",  # Option to return the value of the selection
                    value=list(LabelMode)[0].value
                )

                instruction = gr.Textbox(
                    "Select Area as Label Mask", label="Instruction", interactive=False
                )
                with gr.Row():
                    pos_button = gr.Button("Add mask")
                    neg_button = gr.Button("Remove area")
                with gr.Row():   
                    clear_button = gr.Button("Clear points")
                    undo_button = gr.Button("Undo")

                mask_opacity_slider = gr.Slider(minimum=0, maximum=1, step=0.05, value=0.35, label="Mask Opacity")    
                save_button = gr.Button("Save Current Label & ⏭️ Next Image")

                pre_view_img = gr.Image(label="PreView", show_download_button=False)
                preview_button = gr.Button("PreView Masking Image")

        label_mode_radio.change(self.handle_label_mode, [label_mode_radio, mask_opacity_slider], outputs=[display_view, instruction])
        pos_button.click(self.handle_set_positive, outputs=[instruction])
        neg_button.click(self.handle_set_negative, outputs=[instruction])
        clear_button.click(self.clear_points, [label_mode_radio], outputs=[display_view, instruction])
        undo_button.click(self.undo_points, [label_mode_radio, mask_opacity_slider], outputs=[display_view])


        save_button.click(self.save_mask_event, [label_mode_radio, image_index_slider], outputs=[instruction, image_index_slider])        
        display_view.select(self.get_select_coords, [label_mode_radio, mask_opacity_slider], [display_view])

        mask_opacity_slider.change(self.update_mask_opacity, [label_mode_radio, mask_opacity_slider], [display_view])
        preview_button.click(self.preview_mask_event, [label_mode_radio], outputs=[pre_view_img])

        return display_view, instruction

    def update_img_dir_dropdown(self):
        img_dirs = listdir(self.src_img_dir)

        return  gr.update(choices=img_dirs)

    def select_image_dir(self, img_dir_name):
        select_img_dir = os.path.join(self.src_img_dir, img_dir_name)
        guru.debug(f"Selected image dir: {select_img_dir}")

        self.dst_mask_dir_name = img_dir_name + "_masks"
        self.dst_mask_dir = os.path.join(self.dst_dir_root, self.dst_mask_dir_name)
        return select_img_dir, self.dst_mask_dir_name

    def update_image_dir(self, img_dir):
        self.set_img_dir(img_dir)
        img_index_old = self.img_index_current
        #self.img_index_current = self.get_latest_processed_img_index(img_dir)
        self.img_index_current = self.get_last_img_index(img_dir)

        # enable_force_update ensures the front-end updates if the image index is valid and unchanged,
        # otherwise gr.update triggers a refresh with update_show_img.
        enable_force_update = self.img_index_current > -1 and img_index_old == self.img_index_current
        if enable_force_update:
            display_image = self.update_show_img()
        else:
            display_image = None  # Handle empty directory case

        return gr.update(maximum=len(self.img_paths)- 1, value=self.img_index_current), display_image

    def update_img_index_event(self, img_index):
        self.img_index_current = img_index

        if self.img_index_current > -1:
            display_image = self.update_show_img()
            self.handle_set_positive()
        else:
            display_image = None  # Handle empty directory case

        self.save_last_indice()

        message = "Select Area as Label Mask"
        return display_image, message

    def resize_img(self, image):
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

            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            image = np.copy(image)

        return image

    def update_show_img(self):

        def set_input_image(i: int = 0):
            guru.debug(f"Setting image_index {i} / {len(self.img_paths)}")
            if i < 0 or i >= len(self.img_paths):
                return 
            for prompt in self.prompts_label.values():
                prompt.reset()

            image = iio.imread(self.img_paths[i])
            if image.shape[-1] == 4:
                image = image[:, :, :3]

            
            self.image = image
            self.display_image = self.resize_img(image)
            self.sam_predictor.reset_predictor()
            self.sam_predictor.set_image(image)

        set_input_image(self.img_index_current)

        return self.display_image

    def update_img_index_by_btn(self, current_value, direction):
        if direction == "prev":
            return min(max(current_value - 1, 0), len(self.img_paths)-1)  # Prevent going below 0
        elif direction == "next":
            return min(current_value + 1, len(self.img_paths)-1)

    def save_mask_event(self, label_mode, img_index):
        src_img_path = self.img_paths[self.img_index_current]

        #for prompt in self.prompts_label.values():
        #    prompt.save_mask_to_dir(self.dst_mask_dir, src_img_path,)

        prompt_label = self.prompts_label[label_mode]
        prompt_label.save_mask_to_dir(self.dst_mask_dir, src_img_path)

        '''
        if self.img_index_current > -1:
            self.processed_indices.add(self.img_index_current)
            self.save_processed_indices()
        '''

        message = f"Saved masks to {self.dst_mask_dir}!"

        next_img_index = self.update_img_index_by_btn(img_index, "next")
        return message, next_img_index

    def save_last_indice(self):
        if self.img_dir is not None:
            last_indices_file = os.path.join(self.img_dir, self.last_indice_filename)
            #guru.debug(f"save last_indice_filename: {last_indices_file}")
            with open(last_indices_file, 'w') as f:
                json.dump(self.img_index_current, f)

    def save_processed_indices(self):
        if self.img_dir is not None:
            processed_indices_file = os.path.join(self.img_dir, self.processed_indices_filename)
            guru.debug(f"save processed_indices_file: {processed_indices_file}")
            with open(processed_indices_file, 'w') as f:
                json.dump(list(self.processed_indices), f)

    def preview_mask_event(self, label_mode):
        if self.img_index_current < 0:
            return None

        src_img_path = self.img_paths[self.img_index_current]
        prompt = self.prompts_label[label_mode]

        index_mask = prompt.index_mask
        if index_mask is None:
            return None

        out_img = self.image.copy()
        mask = index_mask > 0
        out_img[~mask] = 0

        return out_img

    def handle_label_mode(self, label_mode: LabelMode, mask_opacity) -> str:
        message = f"Selected Label: {label_mode}"

        prompt_label = self.prompts_label[label_mode]
        index_mask = prompt_label.index_mask

        if index_mask is None:
            return self.display_image, message

        out = self.draw_label_result(label_mode, index_mask, mask_opacity)
        return out, message

    def handle_set_positive(self) -> str:
        for prompt_label in self.prompts_label.values():
            prompt_label.set_positive()

        return "Selecting positive points"

    def handle_set_negative(self) -> str:
        for prompt_label in self.prompts_label.values():
            prompt_label.set_negative()

        return "Selecting negative points"

    def clear_points(self, label_mode) -> Tuple[np.ndarray, str]:
        message = self.prompts_label[label_mode].clear_points()

        return self.display_image, message

    def undo_points(self, label_mode, mask_opacity):
        prompt_label = self.prompts_label[label_mode]
        index_mask = prompt_label.undo_points()

        if index_mask is None:
            return self.display_image

        guru.debug(f"{index_mask.shape=}")
        return self.draw_label_result(label_mode, index_mask, mask_opacity)

    def get_select_coords(self, label_mode, mask_opacity, evt: gr.SelectData): 

        def toSAMCoordinates(x, y):
            h, w = self.display_image.shape[:2]
            ori_h, ori_w = self.image.shape[:2]
            s = max(ori_h, ori_w) / max(h, w)

            x = round(x*s)
            y = round(y*s)

            return x, y


        x = evt.index[1]  # type: ignore
        y = evt.index[0]  # type: ignore
        x, y = toSAMCoordinates(x, y)
        prompt_label = self.prompts_label[label_mode]
        index_mask = prompt_label.add_point(x, y)

        guru.debug(f"{index_mask.shape=}")
        return self.draw_label_result(label_mode, index_mask, mask_opacity)

    def update_mask_opacity(self, label_mode, mask_opacity):
        prompt_label = self.prompts_label[label_mode]
        index_mask = prompt_label.index_mask

        if index_mask is None:
            return self.display_image

        return self.draw_label_result(label_mode, index_mask, mask_opacity)

    def draw_label_result(self, label_mode, index_mask, mask_opacity):

        def toDisPlayCoordinates(points):
            h, w = self.display_image.shape[:2]
            ori_h, ori_w = self.image.shape[:2]
            s = max(h, w) / max(ori_h, ori_w)

            disply_points = [[round(x * s), round(y * s)] for x, y in points]
            return disply_points

        h, w = self.display_image.shape[:2]
        circle_r = min(max(2, int(max(h, w)*0.005)), 50)

        index_mask = cv2.resize(index_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        palette = get_hls_palette(index_mask.max() + 1)
        color_mask = palette[index_mask]

        prompt_label = self.prompts_label[label_mode]
        disply_points = toDisPlayCoordinates(prompt_label.selected_points)

        out_u = compose_img_mask(self.display_image, color_mask, 1.0-mask_opacity)
        out = draw_points(out_u, disply_points, prompt_label.selected_labels, circle_r=circle_r)

        return out

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
        
        return len(self.img_paths)

    def get_last_img_index(self, img_dir):

        last_indices_file = os.path.join(img_dir, self.last_indice_filename)
        
        if os.path.exists(last_indices_file):
            with open(last_indices_file, 'r') as f:
                last_img_index = json.load(f)
        else:
            last_img_index = 0  # Default to 0 if no file exists

        return last_img_index         

    def get_latest_processed_img_index(self, img_dir):

        processed_indices_file = os.path.join(img_dir, self.processed_indices_filename)
        self.load_processed_indices(processed_indices_file)

        num_imgs = len(self.img_paths)
        latest_img_index = 0
        for idx in range(num_imgs):
            if idx not in self.processed_indices:
                latest_img_index = idx  # Set to the first unprocessed image index
                break
        else:
            # If all images are processed, set img_index to the last index
            latest_img_index = num_imgs - 1

        return latest_img_index

    def load_processed_indices(self, processed_indices_file):
        # Load the processed indices from the file if it exists
        if os.path.exists(processed_indices_file):
            with open(processed_indices_file, 'r') as f:
                self.processed_indices = set(json.load(f))
        else:
            self.processed_indices = set()