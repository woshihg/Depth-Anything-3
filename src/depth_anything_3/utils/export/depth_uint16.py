# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import cv2
import numpy as np

from depth_anything_3.specs import Prediction


def export_to_depth_uint16(
    prediction: Prediction,
    export_dir: str,
):
    """
    Export depth maps as 16-bit uint16 PNG files.
    - If prediction.is_metric is True, it saves depth in millimeters (depth * 1000).
    - If prediction.is_metric is False, it normalizes relative depth to 0-65535.
    """
    output_folder = os.path.join(export_dir, "depth_uint16")
    os.makedirs(output_folder, exist_ok=True)

    for idx in range(prediction.depth.shape[0]):
        depth = prediction.depth[idx]
        depth_uint16 = (depth * 1000.0).clip(0, 65535).astype(np.uint16)
        save_path = os.path.join(output_folder, f"{idx:04d}.png")
        cv2.imwrite(save_path, depth_uint16)
