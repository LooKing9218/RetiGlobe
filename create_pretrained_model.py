import torch
from torchvision import transforms
from PIL import Image
import os

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import argparse
import torch
from clip_modules.modeling.model import CLIPRModel
import numpy as np


def get_args_parser():
    parser = argparse.ArgumentParser(
    )
    parser.add_argument(
        "--nb_knn",
        nargs="+",
        type=int,
        help="Number of NN to use. 20 is usually working the best.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="Temperature used in the voting coefficient",
    )
    parser.add_argument(
        "--gather-on-cpu",
        action="store_true",
        help="Whether to gather the train features on cpu, slower"
        "but useful to avoid OOM for large datasets (e.g. ImageNet22k).",
    )
    parser.add_argument('--architecture', default='DinoV2')
    parser.add_argument(
        "--config_file",
        type=str,
        default="dinov2/configs/train/vitl16.yaml",
        help="Model configuration file",
    )

    parser.add_argument(
        "--opts",
        help="Extra configuration options",
        default=[],
        nargs="+",
    )

    parser.add_argument(
        "--output_dir",
        default="ModelSaved",
    )
    parser.add_argument(
        "--batch_size", default=64, help="specify to train model"
    )

    parser.add_argument(
        "--n-per-class-list",
        nargs="+",
        type=int,
        help="Number to take per class",
    )
    parser.add_argument(
        "--n-tries",
        type=int,
        help="Number of tries",
    )
    return parser




def create_model():
    args_parser = get_args_parser()
    args = args_parser.parse_args()
    weight_path = "Path_to_Pretrained_Model"
    DinoCLIP = CLIPRModel(vision_type='DinoV2',
                          from_checkpoint=True, vision_pretrained=True,
                          weights_path=weight_path, R=8, args=args
                          )
    model = DinoCLIP.vision_model.model
    model = model.cuda()

    return model

def extract_feature(model, img_tensor):
    extracted_tensor = model(img_tensor)
    return extracted_tensor.cpu()


if __name__ == "__main__":
    create_model()

