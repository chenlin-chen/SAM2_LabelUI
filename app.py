
import os

if __name__ == "__main__":
    import argparse
    from ui.ui import ImageSegUI
    from ui.ui_select import ImageSelectUI
    from ui.ui_select_2view import ImageSelect2ViewUI


    parser = argparse.ArgumentParser()
    parser.add_argument("--server_name", type=str, default='0.0.0.0')
    parser.add_argument("--port", type=int, default=7890)
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints/sam2.1_hiera_large.pt")
    parser.add_argument("--model_cfg", type=str, default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--data_dir", type=str,  default="/data/SkinParsing")
    parser.add_argument("--src_img_dir_name", type=str,  default="easyPortrait/images")
    parser.add_argument("--output_dir_name", type=str,  default="LabeledMasks")
    parser.add_argument("--ref_img_dir", type=str,  default=None)
    parser.add_argument("--img_name", type=str, default="images")
    parser.add_argument("--mask_name", type=str, default="masks")
    parser.add_argument("--ui_mode", type=str, default="ui_seg")
    args = parser.parse_args()

    # device = "cuda" if torch.cuda.is_available() else "cpu"

    server_name = args.server_name
    port = args.port
    checkpoint_dir = args.checkpoint_dir
    model_cfg = args.model_cfg
    data_dir = args.data_dir
    src_img_dir_name = args.src_img_dir_name
    output_dir_name = args.output_dir_name
    ref_img_dir = args.ref_img_dir
    ui_mode = args.ui_mode
    
    if ui_mode == "ui_seg":
        ui = ImageSegUI(server_name, port,
            checkpoint_dir=checkpoint_dir,
            model_cfg=model_cfg,
            data_dir=data_dir,
            src_img_dir_name=src_img_dir_name,
            output_dir_name=output_dir_name,
            )
    elif ui_mode == "ui_select":
        ui = ImageSelectUI(server_name, port,
            data_dir=data_dir,
            src_img_dir_name=src_img_dir_name,
            output_dir_name=output_dir_name,
        )
    elif ui_mode == "ui_select_2view":
        ui = ImageSelect2ViewUI(server_name, port,
            data_dir=data_dir,
            src_img_dir_name=src_img_dir_name,
            ref_img_dir=ref_img_dir,
            output_dir_name=output_dir_name,
        )
    else:
        raise ValueError(f"Invalid ui_mode: {ui_mode}. Expected 'ui_seg' or 'ui_select'.")

    ui.activate_ui()
