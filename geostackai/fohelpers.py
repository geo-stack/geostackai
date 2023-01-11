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
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from fiftyone import Dataset

# ---- Standard imports
import sys
import json
import os
import os.path as osp

# ---- Third party app
import numpy as np
import fiftyone as fo
from fiftyone import ViewField as F
from fiftyone.utils.coco import _coco_segmentation_to_mask
from fiftyone.utils.iou import compute_ious


def add_pred_to_sample(sample, preds, class_names: dict, label_name: str):
    # https://voxel51.com/docs/fiftyone/tutorials/evaluate_detections.html

    classes = preds['classes']
    boxes = preds['boxes']
    scores = preds['scores']
    masks = preds['masks']
    detections = []

    w = sample['metadata']['width']
    h = sample['metadata']['height']

    for i in range(len(classes)):
        w = sample['metadata']['width']
        h = sample['metadata']['height']

        if masks[i].sum() == 0:
            # Convert to [top-left-x, top-left-y, width, height]
            # in relative coordinates in [0, 1] x [0, 1]
            x1, y1, x2, y2 = boxes[i]
            rel_box = [x1 / w, y1 / h, (x2 - x1) / w, (y2 - y1) / h]

            detection = fo.Detection(
                label=class_names[classes[i]],
                bounding_box=rel_box,
                confidence=scores[i]
                )
        else:
            detection = fo.Detection.from_mask(
                mask=masks[i],
                label=class_names[classes[i]],
                confidence=scores[i]
                )
        detections.append(detection)

    # Save predictions to dataset
    sample[label_name] = fo.Detections(detections=detections)
    sample.save()


def export_to_coco(dataset: Dataset, export_dir: str,
                   export_media: bool = False, prefix: str = '',
                   data_path: str = None, categories_json: str = None):
    """
    Export a fiftyone dataset to the COCO format.
    """
    dataset.export(
        export_dir=export_dir,
        data_path=data_path,
        dataset_type=fo.types.COCODetectionDataset,
        export_media=export_media,
        label_field='ground_truth'
        )

    with open(osp.join(export_dir, 'labels.json'), "rt") as jsonfile:
        json_data = json.load(jsonfile)
    with open(osp.join(export_dir, f'{prefix}labels.json'), 'w') as jsonfile:
        json.dump(json_data, jsonfile, indent=2, separators=(",", ": "))
    os.remove(osp.join(export_dir, 'labels.json'))

    if categories_json is not None:
        # Save a dictionaries with the categories in a json file.
        categories = {c['id']: c['name'] for c in json_data['categories']}
        with open(osp.join(export_dir, categories_json), 'w') as jsonfile:
            json.dump(categories, jsonfile, indent=2, separators=(",", ": "))


def export_to_cvat(dataset, export_dir: str):
    dataset.export(
        export_dir=export_dir,
        data_path=osp.join(export_dir, 'Data'),
        dataset_type=fo.types.COCODetectionDataset,
        export_media=True,
        )

    label_path = osp.join(export_dir, 'labels.json')
    with open(label_path, "rt") as jsonfile:
        json_data = json.load(jsonfile)
    with open(label_path, 'w') as jsonfile:
        json.dump(json_data, jsonfile, indent=2, separators=(",", ": "))


def load_dataset_from_json(dataset_dir, data_dir, dataset_name):
    """Load a FiftyOneDataset fom json."""
    dataset = fo.Dataset.from_dir(
        dataset_dir=dataset_dir,
        dataset_type=fo.types.FiftyOneDataset,
        name=dataset_name)

    # We need to do this ourselves because we do not thrust fiftyone to handle
    # the path sep "/" and "\\" correctlty.
    for sample in dataset:
        filepath = osp.join(data_dir, sample.filepath).replace('\\', '/')
        sample.filepath = filepath
        sample.save()

    return dataset


def save_dataset_to_json(dataset: Dataset, export_dir: str, rel_dir: str):
    """Save a FiftyOneDataset to json."""
    clone = dataset.clone(persistent=False)

    # We need to do this ourselves because fiftyone is not handling
    # the path sep "/" and "\\" correctlty.
    for sample in clone:
        # Fix sample path format (this is required for macOS compatibility).
        filepath = osp.relpath(sample.filepath, rel_dir).replace('\\', '/')
        sample.filepath = filepath
        sample.save()

    clone.export(
        export_dir=export_dir,
        dataset_type=fo.types.FiftyOneDataset,
        export_media=False,
        )

    for filename in ['samples.json', 'metadata.json']:
        _samples_path = osp.join(export_dir, filename)
        with open(_samples_path, "rt") as jsonfile:
            json_data = json.load(jsonfile)
        with open(_samples_path, 'w') as jsonfile:
            json.dump(json_data, jsonfile, indent=2, separators=(",", ": "))

    fo.delete_dataset(clone.name)


def add_inferences_to_dataset(
        dataset: Dataset, infer_json: str, field: str,
        class_names: list[str], load_segmentation: bool = True,
        iou_treshold: float = None, score_treshold: float = None):
    """
    Add inference results to dataset in label.

    Parameters
    ----------
    dataset : Dataset
        The fiftyone dataset to which to add the inferred detections.
    infer_json : str
        The path to the json file containing the data of the
        inferred detections.
    field : str
        The sample field to which to add the inferred detections.
    class_names : list[str]
        The list of class labels of the inferred detections.
    load_segmentation : bool, optional
        Whether to load the segmentation in addition to the bounding
        boxes of the detected object. The default is True.
    iou_treshold : float, optional
        A value between 0 and 1 corresponding to the IOU (Intersection over
        Union) treshold to use when adding inferred detections to samples.
        By default, iou_treshold is None and is ignored. Inferred dections are
        rejected if they share a IOU value above that treshold with an
        existing detection of the same class.
    score_treshold : float, optional
        A value between 0 and 1 corresponding to the minimum score inferred
        detections must have, otherwise they are rejected. By default,
        score_treshold is None and is ignored.
    """
    n_added_total = 0

    # Load inference data.
    with open(infer_json, "rt") as jsonfile:
        infer_data = json.load(jsonfile)

    # Check that the 'filename' sample field exists and add it otherwise.
    for sample in dataset:
        if 'filename' not in sample:
            sample['filename'] = osp.basename(sample['filepath'])
            sample.save()

    sample_count = len(infer_data)
    for k, (img_file, img_infer) in enumerate(infer_data.items()):
        print("\rProcessing sample {} of {}          "
              .format(k + 1, sample_count), end='')

        view = dataset.match(F("filename") == osp.basename(img_file))
        if view.count() == 0:
            continue
        sample = view.first()

        w = sample['metadata']['width']
        h = sample['metadata']['height']

        classes = img_infer['classes']
        boxes = img_infer['boxes']
        scores = img_infer['scores']
        segmentations = img_infer['segmentation']

        if field in sample and sample[field] is not None:
            detections = [d for d in sample[field]['detections']]
        else:
            detections = []

        # Make sure original detections that were identified manually are
        # clearly indicated as such in the dataset.
        for detection in detections:
            if 'inferred' not in detection:
                detection['inferred'] = False

        # We loop over the new detections in descending order of their score.
        argsort = np.argsort(scores)
        for i in reversed(argsort):
            if score_treshold is not None and scores[i] < score_treshold:
                break

            seg = segmentations[i]
            if len(seg) == 0 or load_segmentation is False:
                # Convert to [top-left-x, top-left-y, width, height]
                # in relative coordinates in [0, 1] x [0, 1]
                x1, y1, x2, y2 = boxes[i]
                bbox = [x1 / w, y1 / h, (x2 - x1) / w, (y2 - y1) / h]

                inferred_detection = fo.Detection(
                    label=class_names[classes[i]],
                    bounding_box=bbox,
                    confidence=scores[i])
            else:
                # We want the mask to span the entire image frame.
                bbox = [0, 0, w, h]
                mask = _coco_segmentation_to_mask(seg, bbox, (w, h))

                inferred_detection = fo.Detection.from_mask(
                    mask=mask,
                    label=class_names[classes[i]],
                    confidence=scores[i])

            # Indicate that this detection was inferred from a machine learning
            # object detection model.
            inferred_detection['inferred'] = True

            # We reject the new detection if there is an existing detection
            # of the same class that shares an IOU above the specified
            # treshold.
            if iou_treshold is None:
                detections.append(inferred_detection)
                n_added_total += 1
            else:
                if len(detections):
                    max_ious = np.max(compute_ious(
                        [inferred_detection],
                        detections,
                        classwise=True,
                        use_boxes=True))
                    if max_ious < iou_treshold:
                        detections.append(inferred_detection)
                        n_added_total += 1
                else:
                    detections.append(inferred_detection)
                    n_added_total += 1

        # Save the augmented detection list to the sample.
        sample[field] = fo.Detections(detections=detections)
        sample.save()

    print(f'{n_added_total} inferred detections added to the dataset.')
