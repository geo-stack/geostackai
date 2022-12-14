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

version_info = (0, 1, 0, 'dev0')
__version__ = '.'.join(map(str, version_info))
__project_url__ = "https://github.com/geo-stack/geostackai"


from geostackai.loss import plot_loss, get_loss
from geostackai.fohelpers import (
    save_dataset_to_json, load_dataset_from_json, export_to_coco)
