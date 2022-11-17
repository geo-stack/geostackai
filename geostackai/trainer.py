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

import copy
import torch
from detectron2.engine import DefaultTrainer
from detectron2.data import (
    DatasetMapper, build_detection_test_loader, build_detection_train_loader,
    detection_utils)
import detectron2.data.transforms as T
from detectron2.engine import HookBase
import detectron2.utils.comm as comm
import cv2


def custom_mapper(dataset_dict):
    """
    A custom DatasetMapper to perform data augmentation in Detectron2.

    https://gilberttanner.com/blog/detectron-2-object-detection-with-pytorch/
    """
    dataset_dict = copy.deepcopy(dataset_dict)

    image = cv2.imread(dataset_dict['file_name'])

    # Remove interlacing artifacts in images taken from lower
    # resolution videos
    image = cv2.GaussianBlur(image, (5, 5), 0)

    transform_list = [
        T.RandomBrightness(0.8, 1.8),
        T.RandomContrast(0.6, 1.3),
        T.RandomSaturation(0.8, 1.4),
        T.RandomRotation(angle=[90, 90]),
        T.RandomLighting(0.7),
        T.RandomFlip(prob=0.4, horizontal=False, vertical=True),
    ]
    image, transforms = T.apply_transform_gens(transform_list, image)

    dataset_dict["image"] = torch.as_tensor(
        image.transpose(2, 0, 1).astype("float32"))

    annos = [
        detection_utils.transform_instance_annotations(
            obj, transforms, image.shape[:2])
        for obj in dataset_dict.pop("annotations")
        if obj.get("iscrowd", 0) == 0
        ]
    instances = detection_utils.annotations_to_instances(
        annos, image.shape[:2])

    dataset_dict["instances"] = detection_utils.filter_empty_instances(
        instances)

    return dataset_dict


class ValLossHook(HookBase):
    # https://github.com/facebookresearch/detectron2/issues/810#issuecomment-935713524
    # https://gist.github.com/ortegatron/c0dad15e49c2b74de8bb09a5615d9f6b

    def __init__(self, eval_period, data_loader):
        super().__init__()
        self._eval_period = eval_period
        self._data_loader = data_loader

    def _do_loss_eval(self):
        with torch.no_grad():
            loss_dict_reduced_stack = []
            for data in self._data_loader:
                loss_dict = self.trainer.model(data)
                loss_dict_reduced = {
                    "val_" + k: v.item() for k, v in
                    comm.reduce_dict(loss_dict).items()}
                loss_dict_reduced_stack.append(loss_dict_reduced)

            if comm.is_main_process():
                df = pd.DataFrame(loss_dict_reduced_stack)
                df['val_total_loss'] = df.sum(axis=1)
                mean_values = df.mean(axis=0)

                self.trainer.storage.put_scalars(
                    **mean_values.to_dict())

    def after_step(self):
        """
        After each step calculates the validation loss and adds it
        to the train storage
        """
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            self._do_loss_eval()
        self.trainer.storage.put_scalars(timetest=12)


class Trainer(DefaultTrainer):
    """
    A trainer with validation loss evaluation and data augmentation
    capabilities.
    """

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=custom_mapper)

    def build_hooks(self):
        """
        Overloads build_hooks to add a hook to calculate loss on
        the test set during training.
        """
        hooks = super().build_hooks()
        hooks.insert(-1, ValLossHook(
            eval_period=self.cfg.EVAL_PERIOD,
            model=self.model,
            data_loader=build_detection_test_loader(
                self.cfg,
                self.cfg.DATASETS.TEST[0],
                DatasetMapper(self.cfg, True))
            ))

        return hooks
