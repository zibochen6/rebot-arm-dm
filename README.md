# reBot Arm B601-DM Real-Robot Eval

This entry point adds the Seeed reBot Arm B601-DM follower option for the
GR00T real-robot evaluator.

Install this evaluator environment and the GR00T package:

```shell
uv pip install -e .
uv pip install --no-deps -e ../../../../
```

The Seeed B601-DM LeRobot plugin requires `motorbridge`. If `uv pip install -e .`
cannot fetch it, install the wheel described in the Seeed setup guide before
running the robot.

Example:

```shell
python eval_rebot_arm_dm.py \
  --robot.type=seeed_b601_dm_follower \
  --robot.id=b601_dm_follower \
  --robot.port=/dev/ttyACM4 \
  --robot.can_adapter=damiao \
  --robot.cameras="{ front: {type: opencv, index_or_path: /dev/video0, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: /dev/video2, width: 640, height: 480, fps: 30}}" \
  --policy_host=localhost \
  --policy_port=5555 \
  --lang_instruction="Grab markers and place into pen holder."
```

If your trained checkpoint uses different video modality keys, pass them in the
same order as the checkpoint expects:

```shell
python eval_rebot_arm_dm.py ... --camera_keys up side
```

The default B601-DM policy input mapping matches `/home/seeed/Music/organize_test_tube`:
`single_arm` is the first five values and `gripper` is the sixth value. For
execution, the policy's `gripper` output is routed to the physical `gripper.pos`
motor so the real gripper is commanded.

If you train a checkpoint whose modality uses all six B601 arm joints plus the
physical gripper, override both input and output mappings explicitly:

```shell
python eval_rebot_arm_dm.py ... \
  --policy_state_keys shoulder_pan.pos shoulder_lift.pos elbow_flex.pos wrist_flex.pos wrist_yaw.pos wrist_roll.pos gripper.pos \
  --action_output_keys shoulder_pan.pos shoulder_lift.pos elbow_flex.pos wrist_flex.pos wrist_yaw.pos wrist_roll.pos gripper.pos
```

