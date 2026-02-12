# Tessolo Hand In-Hand Reorientation

In-hand cube reorientation using the Tessolo Delto 5-finger hand, ported from the Shadow Hand direct environment.

## Task

The hand holds a cube and must rotate it to match a randomly sampled goal orientation. A new goal is sampled each time the current one is reached. The episode terminates if the cube falls (distance from spawn > 0.24 m) or time runs out.

## Hand

Tessolo Delto hand with 5 fingers (fingers 1 and 5 are lateral thumbs, fingers 2-4 are the central trio). All 20 joints are actuated; there are no tendons or coupling joints.

| Property | Value |
|----------|-------|
| DOFs | 20 (5 fingers x 4 joints) |
| Fingertips | 5 (`rl_dg_{1..5}_tip`) |
| Joint naming | `rj_dg_{finger}_{joint}`, finger in 1..5, joint in 1..4 |
| Actuator | Implicit, force-driven |
| Stiffness / Damping | 3.0 / 0.1 (all joints) |
| Effort limit | 30.0 Nm |
| Hand init position | (0.0, 0.0, 0.5) |

## Action Space

| Dim | Description |
|-----|-------------|
| 20 | Relative joint position targets in [-1, 1], mapped to joint limits via `scale()` then smoothed with exponential moving average (`act_moving_average`) |

## Observation Spaces

### Full state (`obs_type="full"`, 149 dims)

| Component | Dims | Description |
|-----------|------|-------------|
| `joint_pos` | 20 | Normalized joint positions (unscaled to [-1, 1]) |
| `joint_vel` | 20 | Joint velocities (scaled by `vel_obs_scale=0.2`) |
| `object_pos` | 3 | Object position relative to env origin |
| `object_rot` | 4 | Object quaternion (w, x, y, z) |
| `object_linvel` | 3 | Object linear velocity |
| `object_angvel` | 3 | Object angular velocity (scaled by `vel_obs_scale`) |
| `in_hand_pos` | 3 | Reference position (object default spawn - 0.04 in z) |
| `goal_rot` | 4 | Goal quaternion |
| `rel_goal_rot` | 4 | Relative rotation: `quat_mul(object_rot, quat_conjugate(goal_rot))` |
| `fingertip_pos` | 15 | 5 fingertip positions (3 each) |
| `fingertip_rot` | 20 | 5 fingertip quaternions (4 each) |
| `fingertip_vel` | 30 | 5 fingertip linear + angular velocities (6 each) |
| `actions` | 20 | Previous actions |
| **Total** | **149** | |

### OpenAI reduced (`obs_type="openai"`, 42 dims)

| Component | Dims | Description |
|-----------|------|-------------|
| `fingertip_pos` | 15 | 5 fingertip positions |
| `object_pos` | 3 | Object position |
| `rel_goal_rot` | 4 | Relative goal rotation |
| `actions` | 20 | Previous actions |
| **Total** | **42** | |

### OpenAI asymmetric critic state (179 dims)

Full state observations (149) + fingertip force-torque sensors (5 x 6 = 30) = **179**.

### Vision (`TessoloHandVisionEnv`, 183 dims actor / 206 dims critic)

| Component | Dims | Description |
|-----------|------|-------------|
| Proprio | 132 | Joint pos/vel + in_hand_pos + goal_rot + fingertip pos/rot/vel + actions |
| CNN embedding | 27 | Feature extractor output from 120x120 RGB+depth+segmentation |
| Goal keypoints | 24 | 8 cube corner positions (3 each) for the goal orientation |
| **Actor total** | **183** | |
| Full state | 179 | Full obs (149) + fingertip force-torque (30) |
| CNN embedding | 27 | Same as actor |
| **Critic total** | **206** | |

The CNN (4-layer ConvNet, reused from Shadow Hand) is trained online to regress cube corner keypoints from camera images. Input: 120x120x7 (RGB + depth + segmentation). Output: 27-dim embedding.

## Reward

```
reward = dist_rew + rot_rew + action_penalty + success_bonus + fall_penalty
```

| Term | Formula | Scale |
|------|---------|-------|
| `dist_rew` | `‖object_pos - in_hand_pos‖` | -10.0 |
| `rot_rew` | `1 / (‖rot_dist‖ + 0.1)` | 1.0 |
| `action_penalty` | `Σ(action²)` | -0.0002 |
| `success_bonus` | Flat bonus when `rot_dist < success_tolerance` | +250 |
| `fall_penalty` | Flat penalty when cube falls | 0 (base) / -50 (OpenAI) |

`rot_dist = 2 * arcsin(clamp(‖quat_diff[1:4]‖, max=1.0))` where `quat_diff = object_rot * conj(goal_rot)`.

## Termination

- **Fall**: `‖object_pos - in_hand_pos‖ >= 0.24`
- **Timeout**: episode length >= `episode_length_s / dt` (10s base, 8s OpenAI)
- **Max successes** (OpenAI only): `consecutive_successes >= 50`

## Simulation

| Parameter | Base | OpenAI |
|-----------|------|--------|
| Physics dt | 1/120 s | 1/60 s |
| Decimation | 2 | 3 |
| Control rate | 60 Hz | 20 Hz |
| Episode length | 10 s | 8 s |
| Num envs | 8192 | 8192 |

## Domain Randomization (OpenAI variant only)

| What | Range | Distribution |
|------|-------|--------------|
| Robot friction | static: [0.7, 1.3] | uniform buckets |
| Joint stiffness | [0.75x, 1.5x] | log-uniform scale |
| Joint damping | [0.3x, 3.0x] | log-uniform scale |
| Joint limits | +/- N(0, 0.01) | gaussian add |
| Object friction | static: [0.7, 1.3] | uniform buckets |
| Object mass | [0.5x, 1.5x] | uniform scale |
| Gravity | z += N(0, 0.4) every 36s | gaussian add |
| Action noise | N(0, 0.05) + bias N(0, 0.015) | gaussian |
| Obs noise | N(0, 0.002) + bias N(0, 0.0001) | gaussian |

No tendon randomization (Tessolo has no tendons, unlike Shadow Hand).

## Object

DexCube (0.06 x 0.06 x 0.06 m), density 567 kg/m3. Spawn position: `(0.132, 0.0, 0.6)`.

## Registered Environments

| Task ID | Config | Env Class | Notes |
|---------|--------|-----------|-------|
| `Isaac-Repose-Cube-Tessolo-Direct-v0` | `TessoloHandEnvCfg` | `InHandManipulationEnv` | Full obs, no DR |
| `Isaac-Repose-Cube-Tessolo-OpenAI-FF-Direct-v0` | `TessoloHandOpenAIEnvCfg` | `InHandManipulationEnv` | Reduced obs, asymmetric critic, DR |
| `Isaac-Repose-Cube-Tessolo-OpenAI-LSTM-Direct-v0` | `TessoloHandOpenAIEnvCfg` | `InHandManipulationEnv` | Same config, LSTM policy |
| `Isaac-Repose-Cube-Tessolo-Vision-Direct-v0` | `TessoloHandVisionEnvCfg` | `TessoloHandVisionEnv` | Camera + CNN, 1225 envs |
| `Isaac-Repose-Cube-Tessolo-Vision-Direct-Play-v0` | `TessoloHandVisionEnvPlayCfg` | `TessoloHandVisionEnv` | Inference mode, 64 envs |

## Commands

```bash
# Zero agent
python3 scripts/environments/zero_agent.py --task Isaac-Repose-Cube-Tessolo-Direct-v0 --num_envs 1

# Random agent
python3 scripts/environments/random_agent.py --task Isaac-Repose-Cube-Tessolo-Direct-v0 --num_envs 1

# Train (rl_games)
python3 scripts/reinforcement_learning/rl_games/train.py \
    --task Isaac-Repose-Cube-Tessolo-Direct-v0 --num_envs 8192 --headless

# Train OpenAI FF
python3 scripts/reinforcement_learning/rl_games/train.py \
    --task Isaac-Repose-Cube-Tessolo-OpenAI-FF-Direct-v0 --num_envs 8192 --headless

# Train vision (requires cameras)
python3 scripts/reinforcement_learning/rl_games/train.py \
    --task Isaac-Repose-Cube-Tessolo-Vision-Direct-v0 --num_envs 1225 --headless --enable_cameras

# Play
python3 scripts/reinforcement_learning/rl_games/play.py \
    --task Isaac-Repose-Cube-Tessolo-Direct-v0 --num_envs 64 \
    --checkpoint logs/rl_games/tessolo_hand/<run_id>/nn/<checkpoint>.pth
```

## Differences from Shadow Hand

| Aspect | Shadow Hand | Tessolo |
|--------|-------------|---------|
| Total joints | 24 (20 actuated + 4 coupling) | 20 (all actuated) |
| Tendons | Yes (fixed tendons) | No |
| Full obs dim | 157 | 149 |
| State dim | 187 | 179 |
| Vision obs dim | 191 | 183 |
| Vision state dim | 214 | 206 |
| Tendon DR | Yes | Removed |
| Finger layout | 4 fingers + 1 thumb | 3 central + 2 lateral thumbs |
