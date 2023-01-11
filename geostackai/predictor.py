# -*- coding: utf-8 -*-
# =============================================================================
# Copyright (C) Les solutions géostack, Inc - All Rights Reserved
#
# This file is part of geostackai
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
#
# https://github.com/geo-stack/geostackai
# =============================================================================
from __future__ import annotations

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['KMP_DUPLICATE_LIB_OK'] = "True"

# ---- Standard imports
import cv2
import requests

# ---- Setup detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
import detectron2.data.transforms as T

# ---- Third party imports
import torch
import numpy as np

# ---- Local imports
from geostackai.trainer import BASE_HEIGHT


def format_outputs(prediction):
    """Return a json serializable version of outputs."""
    data_out = {}
    data_out['classes'] = ([
        int(out) for out in prediction['instances'].pred_classes.cpu().numpy()
        ])

    data_out['boxes'] = ([
        list(out.astype(float)) for out in
        prediction['instances'].pred_boxes.tensor.cpu().numpy()
        ])
    data_out['scores'] = (
        list(prediction['instances'].scores.cpu().numpy().astype(float))
        )
    try:
        data_out['masks'] = prediction['instances'].pred_masks.cpu().numpy()
    except AttributeError:
        pass
    return data_out


def decimate_contour(contour, max_decimation):
    decimation = len(contour)//20
    if decimation < 10:
        return contour
    else:
        if decimation % 2 != 0:
            decimation -= 1
        if decimation > max_decimation:
            decimation = max_decimation

        idx = np.arange(0, len(contour), decimation)

        # Add last point.
        idx = np.append(idx, len(contour) - 1)
        return contour[idx]


def mask_to_segmentation(binary_mask):
    ret, thresh = cv2.threshold(binary_mask.astype(np.uint8), 0.5, 1, 0)
    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    segmentation = []
    for i, contour in enumerate(contours):
        contour = np.flip(contour, axis=1)
        if len(contour) >= 10:
            contour = decimate_contour(contour, 50)
            segmentation.append(contour.ravel().tolist())
    return segmentation


class Predictor(DefaultPredictor):

    def __init__(self, options):

        cfg = get_cfg()

        # https://github.com/facebookresearch/detectron2/issues/2082
        cfg.set_new_allowed(True)

        try:
            config_file = options.config_file
        except AttributeError:
            config_file = model_zoo.get_config_file(
                "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

        cfg.merge_from_file(config_file)
        try:
            cfg.MODEL.ROI_HEADS.NUM_CLASSES = options.num_classes
        except AttributeError:
            pass

        cfg.MODEL.WEIGHTS = options.model
        cfg.SOLVER.IMS_PER_BATCH = 1

        if torch.cuda.is_available():
            print(' * GPU calculations available.')
            cfg.MODEL.DEVICE = "cuda"
        else:
            print(' * GPU calculations are unavailable on this device: '
                  'calculations will be performed on CPU instead.')

        # Set the testing threshold for this model.
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = options.treshold

        super().__init__(cfg)

    def predict(self, image: str | np.ndarray):
        if not isinstance(image, np.ndarray):
            if image.startswith(('http', 'https')):
                image_reponse = requests.get(image)
                image_as_np_array = np.frombuffer(
                    image_reponse.content, np.uint8)
                image = cv2.imdecode(image_as_np_array, cv2.IMREAD_COLOR)
            else:
                image = cv2.imread(image)
        else:
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        # Calcul scale to height new image size.
        orig_height = image.shape[0]
        orig_width = image.shape[1]

        vscale_factor = BASE_HEIGHT / orig_height
        new_height = int(image.shape[0] * vscale_factor)
        new_width = int(image.shape[1] * vscale_factor)

        # Apply transformations to the image.
        image, transforms = T.apply_transform_gens(
            [T.Resize((new_height, new_width))], image)

        # Smooth image to remove interlacing artifacts that are
        # present in images taken from lower resolution videos
        image = cv2.GaussianBlur(image, (3, 3), 0)

        # Do the prediction.
        outputs = self(image)

        prediction = format_outputs(outputs)

        # Transform the predictions back to the original image size.
        transform = T.Resize((orig_height, orig_width)).get_transform(image)

        prediction['masks'] = [
            transform.apply_segmentation(mask.astype(np.uint8))
            for mask in prediction['masks']
            ]
        prediction['boxes'] = [
            transform.apply_coords(
                np.array(bbox).reshape(-1, 2)).flatten().tolist()
            for bbox in prediction['boxes']
            ]
        prediction['segmentation'] = ([
            list(mask_to_segmentation(mask.astype(float))) for
            mask in prediction['masks']
            ])

        return prediction


if __name__ == "__main__":
    from argparse import Namespace
    import json
    import os.path as osp
    import matplotlib.pyplot as plt
    plt.close('all')

    dirpath = (
        "G:/Shared drives/2_PROJETS/211209_CTSpec_AI_inspection_conduites/"
        "2_TECHNIQUE/6_TRAITEMENT/1_DATA/Training/")

    json_filepath = osp.join(dirpath, 'Models', 'train_labels.json')
    model_pth = osp.join(dirpath, 'Models', 'testMarty', 'model_0004999.pth')
    config_yaml = osp.join(dirpath, 'Models', 'testMarty', 'configs.yaml')
    data_path = osp.join(dirpath, 'Data')

    # Define the list of class names.
    with open(json_filepath, "rt") as jsonfile:
        json_data = json.load(jsonfile)
    class_names = [cat['name'] for cat in json_data['categories']]

    options = Namespace(
        model=model_pth,
        num_classes=len(class_names),
        treshold=0.2,
        config_file=model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        )
    predictor = Predictor(options)
