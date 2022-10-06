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

# ---- Standard imports
import sys
import json
import os
import os.path as osp

# We need to import 'annotations' from '__future__' for python versions < 3.9.

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


def load_dataset_from_json(dataset_dir, data_dir, dataset_name):
    return fo.Dataset.from_dir(
        dataset_dir=dataset_dir,
        rel_dir=data_dir,
        dataset_type=fo.types.FiftyOneDataset,
        name=dataset_name)


def save_dataset_to_json(export_dir, dataset):
    rel_dir = dataset.first()['filepath']
    dataset.export(
        export_dir=export_dir,
        dataset_type=fo.types.FiftyOneDataset,
        export_media=False,
        rel_dir=rel_dir
        )

    for filename in ['samples.json', 'metadata.json']:
        _samples_path = osp.join(export_dir, filename)
        with open(_samples_path, "rt") as jsonfile:
            json_data = json.load(jsonfile)
        with open(_samples_path, 'w') as jsonfile:
            json.dump(json_data, jsonfile, indent=2, separators=(",", ": "))


if __name__ == '__main__':
    pass
