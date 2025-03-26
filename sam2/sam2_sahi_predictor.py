from typing import Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from cv2 import Mat
from PIL import Image
import math

from sam2.sam2_image_predictor import SAM2ImagePredictor
from sahi.slicing import slice_coco
from sahi.utils.file import load_json

from sam2.build_sam import build_sam2
from sam2.utils.data_utils import load_image

class SAHIpredictor:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    
    def calculate_slices(self, w, h, w1, h1, overlap_percent):
        a = (overlap_percent / 100) * w1
        n_cols = math.ceil((w - a) / (w1 - a))
        n_rows = math.ceil((h - a) / (h1 - a))
        return n_cols, n_rows

    def load_boxes(file_path):
        pass

    def image_predict(
            self,
            source,
            label,
            model_ckpt,
            model_cfg,
            input_box=None,
            input_point=None,
            input_label=None,
            multimask_output=False,
    ):
        image = load_image(source)
        sam2_model = build_sam2(model_cfg, model_ckpt, device=self.device)
        predictor = SAM2ImagePredictor(sam2_model)
        predictor.set_image(image)

        if input_box is None:
            # input_box = self.load_boxes(label)
            f = load_json(label)
            input_box = f["annotations"][42]['bbox']


        masks, scores, _ = predictor.predict(point_coords=input_point,
                                             point_labels=input_label,
                                             box=input_box,
                                             multimask_output=multimask_output)
        

        
        return masks, scores, _
    
    def slicing_predict(
        self,
        file_path,
    ):
        f = load_json(file_path)


    
