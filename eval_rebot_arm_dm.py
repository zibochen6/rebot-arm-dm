# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Seeed reBot Arm B601-DM real-robot GR00T policy evaluation.

This entry point is intentionally standalone because the B601-DM arm has
different joints and a plugin-provided LeRobot robot class.
"""

from dataclasses import asdict, dataclass
import importlib
import logging
from pprint import pformat
import signal
import sys
import time
from typing import Any

import draccus
from gr00t.policy.server_client import PolicyClient
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.robots import RobotConfig, make_robot_from_config
try:
    from lerobot.utils.import_utils import register_third_party_plugins
except ImportError:
    from lerobot.utils.import_utils import register_third_party_devices as register_third_party_plugins
from lerobot.utils.utils import init_logging, log_say
import numpy as np


for module_name in ["lerobot_robot_seeed_b601"]:
    try:
        importlib.import_module(module_name)
    except ImportError:
        pass


ROBOT_TYPE = "seeed_b601_dm_follower"
ROBOT_STATE_KEYS = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_yaw.pos",
    "wrist_roll.pos",
    "gripper.pos",
]

# Matches B601-DM datasets with six arm joints plus one physical gripper:
# state/action.single_arm = [0:6], state/action.gripper = [6:7].
DEFAULT_POLICY_STATE_KEYS = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_yaw.pos",
    "wrist_roll.pos",
    "gripper.pos",
]

# Map the policy's 6-D single_arm output to the six B601-DM arm joints and
# the policy's gripper output to the physical gripper motor.
DEFAULT_ACTION_OUTPUT_KEYS = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_yaw.pos",
    "wrist_roll.pos",
    "gripper.pos",
]

DEFAULT_CAMERA_KEYS = ["front", "side"]


def recursive_add_extra_dim(obs: dict) -> dict:
    for key, val in obs.items():
        if isinstance(val, np.ndarray):
            obs[key] = val[np.newaxis, ...]
        elif isinstance(val, dict):
            obs[key] = recursive_add_extra_dim(val)
        else:
            obs[key] = [val]
    return obs


class RebotArmDMAdapter:
    def __init__(
        self,
        policy_client: PolicyClient,
        camera_keys: list[str] | None = None,
        policy_state_keys: list[str] | None = None,
        action_output_keys: list[str] | None = None,
    ):
        self.policy = policy_client
        self.camera_keys = camera_keys or DEFAULT_CAMERA_KEYS
        self.policy_state_keys = policy_state_keys or DEFAULT_POLICY_STATE_KEYS
        self.action_output_keys = action_output_keys or DEFAULT_ACTION_OUTPUT_KEYS

        self._validate_keys("policy_state_keys", self.policy_state_keys)
        self._validate_keys("action_output_keys", self.action_output_keys)
        if len(self.policy_state_keys) < 2:
            raise ValueError("policy_state_keys must include arm keys plus one gripper key")
        if len(self.action_output_keys) != len(self.policy_state_keys):
            raise ValueError(
                "action_output_keys must have the same length as policy_state_keys "
                "so policy outputs can be mapped unambiguously"
            )

        self.arm_dof = len(self.policy_state_keys) - 1
        self._prev_action: dict[str, float] | None = None
        self._smoothing_alpha: float = 0.7
        self._smoothing_max_delta: float = 5.0
        self._smoothing_gripper_alpha: float = 0.3

    def init_smoothing(self, current_state: dict[str, float]) -> None:
        """Initialize the smoothing filter with the robot's current joint positions.

        Call this once before the first action is sent so the EMA has a correct
        baseline instead of starting from a raw policy output.
        """
        self._prev_action = {key: current_state[key] for key in self.action_output_keys}

    @staticmethod
    def _validate_keys(field_name: str, keys: list[str]) -> None:
        unknown = [key for key in keys if key not in ROBOT_STATE_KEYS]
        if unknown:
            raise ValueError(f"{field_name} contains unsupported B601-DM keys: {unknown}")

    def obs_to_policy_inputs(self, obs: dict[str, Any]) -> dict:
        missing_camera_keys = [key for key in self.camera_keys if key not in obs]
        if missing_camera_keys:
            raise KeyError(
                f"Robot observation is missing camera keys {missing_camera_keys}. "
                f"Available keys: {sorted(obs.keys())}"
            )

        missing_state_keys = [key for key in self.policy_state_keys if key not in obs]
        if missing_state_keys:
            raise KeyError(
                f"Robot observation is missing state keys {missing_state_keys}. "
                f"Available keys: {sorted(obs.keys())}"
            )

        state = np.array([obs[key] for key in self.policy_state_keys], dtype=np.float32)
        model_obs = {
            "video": {key: obs[key] for key in self.camera_keys},
            "state": {
                "single_arm": state[: self.arm_dof],
                "gripper": state[self.arm_dof : self.arm_dof + 1],
            },
            "language": {"annotation.human.task_description": obs["lang"]},
        }
        model_obs = recursive_add_extra_dim(model_obs)
        model_obs = recursive_add_extra_dim(model_obs)
        return model_obs

    def decode_action_chunk(self, chunk: dict, t: int) -> dict[str, float]:
        single_arm = chunk["single_arm"][0][t]
        gripper = chunk["gripper"][0][t]
        if single_arm.shape[-1] != self.arm_dof:
            raise ValueError(
                f"Policy returned single_arm dim {single_arm.shape[-1]}, expected {self.arm_dof}. "
                "Use policy_state_keys/action_output_keys that match the checkpoint modality."
            )

        full = np.concatenate([single_arm, gripper], axis=0)
        return {key: float(full[i]) for i, key in enumerate(self.action_output_keys)}

    def get_action(self, obs: dict) -> list[dict[str, float]]:
        model_input = self.obs_to_policy_inputs(obs)
        action_chunk, info = self.policy.get_action(model_input)
        any_key = next(iter(action_chunk.keys()))
        horizon = action_chunk[any_key].shape[1]
        return [self.decode_action_chunk(action_chunk, t) for t in range(horizon)]

    def set_smoothing(self, alpha: float, max_delta: float, gripper_alpha: float = 0.3):
        """Configure EMA smoothing parameters.

        Args:
            alpha: EMA smoothing factor for arm joints in (0, 1]. Higher = less smoothing.
                0.7 means 70% new action + 30% previous filtered action.
            max_delta: Maximum allowed joint change per step in degrees.
                Actions with delta > max_delta are clamped. Set to None to disable.
            gripper_alpha: EMA smoothing factor for the gripper. Lower values reduce
                oscillation. 0.3 gives 30% new + 70% previous (smoother than arm).
        """
        self._smoothing_alpha = alpha
        self._smoothing_max_delta = max_delta
        self._smoothing_gripper_alpha = gripper_alpha

    def _smooth_actions(
        self, actions: list[dict[str, float]]
    ) -> list[dict[str, float]]:
        """
        Apply EMA smoothing + per-step delta clamping across the entire action horizon.

        Each action in the batch is smoothed against the *previous filtered action*
        (not the raw policy output), so the arm moves smoothly even within a
        single batch.  Gripper actions use a gentler alpha to avoid oscillation
        while still tracking open/close intent.
        """
        if self._prev_action is None:
            raise RuntimeError(
                "init_smoothing() must be called with the robot's current state "
                "before the first action is sent."
            )

        smoothed = []
        for action_dict in actions:
            filtered = {}
            for key, value in action_dict.items():
                prev = self._prev_action[key]
                is_gripper = key == "gripper.pos"

                # Step 1: clamp delta per step
                delta = value - prev
                if abs(delta) > self._smoothing_max_delta:
                    value = prev + (self._smoothing_max_delta if delta > 0 else -self._smoothing_max_delta)

                # Step 2: EMA smoothing — gripper uses a gentler alpha
                alpha = self._smoothing_gripper_alpha if is_gripper else self._smoothing_alpha
                filtered[key] = alpha * value + (1 - alpha) * prev

            smoothed.append(filtered)
            self._prev_action = filtered

        return smoothed


@dataclass
class EvalConfig:
    robot: RobotConfig
    policy_host: str = "localhost"
    policy_port: int = 5555
    action_horizon: int = 8
    lang_instruction: str = "Grab markers and place into pen holder."
    camera_keys: list[str] | None = None
    policy_state_keys: list[str] | None = None
    action_output_keys: list[str] | None = None
    play_sounds: bool = False
    timeout: int = 60
    action_smoothing_alpha: float = 0.7
    action_smoothing_max_delta: float = 5.0
    action_smoothing_gripper_alpha: float = 0.3


@draccus.wrap()
def eval(cfg: EvalConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    if cfg.robot.type != ROBOT_TYPE:
        raise ValueError(f"eval_rebot_arm_dm.py only supports --robot.type={ROBOT_TYPE}")

    robot = make_robot_from_config(cfg.robot)
    robot.connect()

    policy = None  # type: RebotArmDMAdapter | None

    def _cleanup(sig, frame):
        logging.info("Ctrl+C received — sending zero action and disabling robot...")
        try:
            # If policy is available, use its action keys; otherwise fall back to ROBOT_STATE_KEYS
            action_keys = ROBOT_STATE_KEYS if policy is None else policy.action_output_keys
            robot.send_action({key: 0.0 for key in action_keys})
            time.sleep(0.5)
            robot.disconnect()
            logging.info("Robot zeroed and disconnected.")
        except Exception:
            pass
        sys.exit(0)

    signal.signal(signal.SIGINT, _cleanup)
    signal.signal(signal.SIGTERM, _cleanup)

    log_say("Initializing robot", cfg.play_sounds, blocking=True)

    policy_client = PolicyClient(host=cfg.policy_host, port=cfg.policy_port)
    policy = RebotArmDMAdapter(
        policy_client,
        cfg.camera_keys,
        cfg.policy_state_keys,
        cfg.action_output_keys,
    )
    policy.set_smoothing(cfg.action_smoothing_alpha, cfg.action_smoothing_max_delta, cfg.action_smoothing_gripper_alpha)

    # Get initial observation and seed the smoother with actual joint positions
    # so the first smoothed action is a valid delta from where the arm already is.
    init_obs = robot.get_observation()
    init_state = {key: float(init_obs[key]) for key in policy.action_output_keys}
    policy.init_smoothing(init_state)

    log_say(
        f'Policy ready with instruction: "{cfg.lang_instruction}"',
        cfg.play_sounds,
        blocking=True,
    )

    while True:
        obs = robot.get_observation()
        obs["lang"] = cfg.lang_instruction

        actions = policy.get_action(obs)
        smoothed_actions = policy._smooth_actions(actions)
        for i, action_dict in enumerate(smoothed_actions[: cfg.action_horizon]):
            tic = time.time()
            print(f"action[{i}]: {action_dict}")
            robot.send_action(action_dict)
            toc = time.time()
            if toc - tic < 1.0 / 30:
                time.sleep(1.0 / 30 - (toc - tic))


def main():
    register_third_party_plugins()
    eval()


if __name__ == "__main__":
    main()

