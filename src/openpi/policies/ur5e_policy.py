import dataclasses
import logging
import os
import pathlib
import threading
import time

import einops
import imageio.v2 as imageio
import numpy as np

from openpi import transforms
from openpi.models import model as _model

logger = logging.getLogger("openpi")

_TARGET_IMAGE_HEIGHT = 224
_TARGET_IMAGE_WIDTH = 224


def make_ur5e_example() -> dict:
    """Creates a random input example for the UR5E policy."""
    return {
        "observation/exterior_image_1_left": np.random.randint(
            256, size=(_TARGET_IMAGE_HEIGHT, _TARGET_IMAGE_WIDTH, 3), dtype=np.uint8
        ),
        "observation/wrist_image_left": np.random.randint(
            256, size=(_TARGET_IMAGE_HEIGHT, _TARGET_IMAGE_WIDTH, 3), dtype=np.uint8
        ),
        "observation/joint_position": np.random.rand(6),
        "observation/gripper_position": np.random.rand(1),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class UR5Inputs(transforms.DataTransformFn):
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        """
        Args:
            data (dict): Dictionary containing:
                - "observation/joints": np.ndarray of shape (6,) / (7,) representing joint positions (+ gripper)
                - "observation/base_rgb": np.ndarray representing base camera image
                - "observation/wrist_rgb": np.ndarray representing wrist camera image
                - Optional keys:
                    - "actions": np.ndarray of shape (T, 7) representing action sequences
                    - "prompt": str representing language instruction
                    - "observation/gripper_position": np.ndarray of shape (1,) representing gripper state

        Returns:
            dict: _description_
        """

        joints = np.asarray(data["observation/joints"])

        if joints.ndim != 1:
            raise ValueError(f"Expected joints to be 1D, got shape {joints.shape}")

        if "observation/gripper_position" in data:
            gripper_pos = np.asarray(data["observation/gripper_position"])
            if gripper_pos.ndim == 0:
                # Ensure gripper position is a 1D array, not a scalar, so we can concatenate with joint positions
                gripper_pos = gripper_pos[np.newaxis]
            state = np.concatenate([joints, gripper_pos])
        # Gripper position is embedded in the state or missing.
        elif joints.shape[-1] == 7:
            state = joints
        elif joints.shape[-1] == 6:
            # Append zero placeholder for missing gripper value.
            state = np.concatenate([joints, np.zeros(1, dtype=joints.dtype)])
        else:
            raise ValueError(f"Unexpected joint/state shape: {joints.shape}")

        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference.
        base_image = _parse_image(data["observation/base_rgb"])
        wrist_image = _parse_image(data["observation/wrist_rgb"])

        # Create inputs dict.
        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                # Since there is no right wrist, replace with zeros
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # Since the "slot" for the right wrist is not used, this mask is set
                # to False
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])

        # Pass the prompt (aka language instruction) to the model.
        if "prompt" in data:
            if isinstance(data["prompt"], bytes):
                data["prompt"] = data["prompt"].decode("utf-8")
            inputs["prompt"] = data["prompt"]

        logger.debug(f"Inputs: {inputs}")

        return inputs


@dataclasses.dataclass(frozen=True)
class UR5Outputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # Since the robot has 7 action dimensions (6 DoF + gripper), return the first 7 dims
        return {"actions": np.asarray(data["actions"][:, :7])}
