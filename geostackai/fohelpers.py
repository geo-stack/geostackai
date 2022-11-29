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
import fiftyone as fo


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


if __name__ == '__main__':
    pass
