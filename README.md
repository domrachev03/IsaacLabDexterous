![Isaac Lab](docs/source/_static/isaaclab.jpg)

---

# Dexsuite for Isaac Lab

Dexsuite is a dexterous manipulation suite built on Isaac Lab. A single base dexsuite setup defines common MDPs, rewards,
observations, and assets, and multiple robot-specific configurations plug into it (UR10 Tessolo, Kuka Allegro, Panda RoHand, and more).
All robot variants share the same control, logging, and training pipeline while swapping only the robot/hand mixins and task configs.

## Branches
- `feat/panda_rohand_ang`: adds a challenging setup with a 7-DoF Panda arm and a 6-DoF RoHand; each finger uses a 4-bar linkage and angle-based actuation.
- `feat/visible_points_only`: keeps the same dexsuite base but exposes variants where perception/inputs can be limited to visible points only.
Both branches use the same commands below; just `git checkout` the branch you need and select the corresponding task id for that robot setup.

## Base layout (shared across robot setups)
The dexsuite code is located in `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/dexsuite/` and consists of:
- `dexsuite_env_cfg.py`: base dexsuite environment definitions (reorient and lift tasks).
- `mdp/*`: shared MDP pieces (observations, terminations, rewards).
- `config/`: robot overlays:
  - `ur10_tessolo/`: UR10 + Tessolo mixins. 
    - `dexsuite_ur10_tessolo_env_cfg.py`: UR10 + Tessolo mixins (assets, control, observations) layered on top of the base dexsuite reorient/lift env cfgs.
    - `agents/rl_games_ppo_cfg.yaml`: PPO hyperparameters for `rl_games` on UR10 + Tessolo.
  - `kuka_allegro/`: Kuka + Allegro mixins.
- `panda_rohand/`: Panda + RoHand mixins (available on `feat/panda_rohand_ang`).

### Visible-points-only branch (`feat/visible_points_only`)
- Observation change: uses `mdp.visible_object_point_cloud_b` to sample object surface points once, project them into the RGB-D camera, and keep only points passing depth + instance segmentation checks. Requires cameras to be enabled at runtime.
- Extra task IDs (UR10 + Tessolo): `Isaac-Dexsuite-UR10-Tessolo-Reorient-Visible-v0`, `Isaac-Dexsuite-UR10-Tessolo-Reorient-Visible-Play-v0`, `Isaac-Dexsuite-UR10-Tessolo-Lift-Visible-v0`, `Isaac-Dexsuite-UR10-Tessolo-Lift-Visible-Play-v0` (mirrored for Kuka + Allegro).
- Agent configs: `config/*/agents/rl_games_ppo_visible_cfg.yaml` (experiment names like `reorient_visible`).
- Commands (append `--enable_cameras`):
  - Zero: `python3 scripts/environments/zero_agent.py --task Isaac-Dexsuite-UR10-Tessolo-Reorient-Visible-v0 --num_envs 1 --enable_cameras`
  - Random: `python3 scripts/environments/random_agent.py --task Isaac-Dexsuite-UR10-Tessolo-Reorient-Visible-v0 --num_envs 1 --enable_cameras`
  - Teleop: `python3 scripts/environments/teleoperation/teleop_se3_agent.py --task Isaac-Dexsuite-UR10-Tessolo-Reorient-Visible-Play-v0 --num_envs 1 --teleop_device keyboard --enable_cameras`
  - Training: `python3 scripts/reinforcement_learning/rl_games/train.py --task Isaac-Dexsuite-UR10-Tessolo-Reorient-Visible-v0 --num_envs 4096 --headless --enable_cameras --wandb-project-name UR10TesolloVisionBased --wandb-entity cs672_team --wandb-name UR-Tessolo-Vision --track`
  - Play: `python3 scripts/reinforcement_learning/rl_games/play.py --task Isaac-Dexsuite-UR10-Tessolo-Reorient-Visible-v0 --num_envs 512 --checkpoint <path_to_checkpoint> --enable_cameras`

### W&B logging
WandB logging allows you to track the progress and compare different runs. Before using, you need to login to your WandB account:
```bash
wandb login
```
Here is information about the W&B setup used in this project:
- Entity: `cs672_team`.
- Projects: [UR10TesolloVisionBased](https://wandb.ai/cs672_team/UR10TesolloVisionBased) (UR10 Tessolo state/vision runs) and [panda_rohand_lift](https://wandb.ai/cs672_team/panda_rohand_lift) (Panda + RoHand runs).
- Add to runs: `--wandb-entity cs672_team --wandb-project-name <project> --wandb-name <run_name> --track`.

### Installation Instructions
Below we provide venv/conda-based installation (works for `main` and for the two feature branches). 
1. Create any python virtual env (venv, conda env, uv env)
2. Install PyTorch. Update cuda version in `index-url` if needed:
```bash
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
```
3. Install `rl_games`, which we primarily use for RL training:
```bash
pip install  git+https://github.com/isaac-sim/rl_games.git@python3.11
```
4. Install `isaacsim`, which is the core simulation engine:
```bash
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
```
5. Clone and enter Isaac Lab repository:
```bash
git clone https://github.com/domrachev03/IsaacLabDexterous.git
cd IsaacLabDexterous
# optional: git checkout feat/panda_rohand_ang    # Panda + RoHand setup
# optional: git checkout feat/visible_points_only # visible-points-only observations
```
6. Install Isaac Lab:
```bash
./isaaclab.sh -i
```

### Environments
One base dexsuite environment backs multiple robot setups:
- UR10 + Tessolo hand (reorient/lift; `Isaac-Dexsuite-UR10-Tessolo-*-v0`) on all branches
- Kuka + Allegro hand (reorient/lift; `Isaac-Dexsuite-Kuka-Allegro-*-v0`) on all branches
- Panda + RoHand (reorient/lift; available after `git checkout feat/panda_rohand_ang`) with 7-DoF arm and 4-bar linked fingers.
- Visible-only perception variants exist on `feat/visible_points_only` (see task IDs above) and require `--enable_cameras`.

### Running Environment
Use the same scripts on any branch; switch the `--task` to the robot you want:
1. Zero Agent -- spawns environment with zero actions, e.g.:
```bash
python3 scripts/environments/zero_agent.py --task Isaac-Dexsuite-UR10-Tessolo-Lift-v0 --num_envs 1
```
2. Random Agent -- spawns environment with random actions, e.g.:
```bash
python3 scripts/environments/random_agent.py --task Isaac-Dexsuite-UR10-Tessolo-Lift-v0 --num_envs 1
```
3. Teleoperation -- control with keyboard/gamepad on a teleop-friendly variant:
```bash
python3 scripts/environments/teleoperation/teleop_se3_agent.py --task Isaac-Dexsuite-UR10-Tessolo-Lift-Play-v0 --num_envs 1 --teleop_device keyboard
```
Replace the task id with any other dexsuite robot; for visible-point tasks also add `--enable_cameras`.

### Training
To train with `rl_games`, include W&B flags:
- UR10 Tessolo (state-based):
```bash
python3 scripts/reinforcement_learning/rl_games/train.py --task Isaac-Dexsuite-UR10-Tessolo-Lift-v0 --num_envs 4096 --headless --wandb-project-name UR10TesolloVisionBased --wandb-entity cs672_team --wandb-name UR-Tessolo-State --track
```
- UR10 Tessolo vision-based (visible-point env on `feat/visible_points_only`, requires cameras):
```bash
python3 scripts/reinforcement_learning/rl_games/train.py --task Isaac-Dexsuite-UR10-Tessolo-Lift-Visible-v0 --num_envs 4096 --headless --enable_cameras --wandb-project-name UR10TesolloVisionBased --wandb-entity cs672_team --wandb-name UR-Tessolo-Vision --track
```
- Panda + RoHand (lift example on `feat/panda_rohand_ang`):
```bash
python3 scripts/reinforcement_learning/rl_games/train.py --task Isaac-Dexsuite-Panda-RoHand-Lift-v0 --num_envs 4096 --headless --wandb-project-name panda_rohand_lift --wandb-entity cs672_team --wandb-name Panda-RoHand-Lift --track
```

To run a trained policy, use the corresponding `play.py` script and point `--checkpoint` to the run you trained (logs land in `logs/rl_games/<config_name>/<run_id>/nn/`):
```bash
python3 scripts/reinforcement_learning/rl_games/play.py --task Isaac-Dexsuite-UR10-Tessolo-Lift-v0 --num_envs 512 --checkpoint logs/rl_games/lift/<run_id>/nn/<checkpoint>.pth
```


## License

The Isaac Lab framework is released under [BSD-3 License](LICENSE). The `isaaclab_mimic` extension and its
corresponding standalone scripts are released under [Apache 2.0](LICENSE-mimic). The license files of its
dependencies and assets are present in the [`docs/licenses`](docs/licenses) directory.

Note that Isaac Lab requires Isaac Sim, which includes components under proprietary licensing terms. Please see the [Isaac Sim license](docs/licenses/dependencies/isaacsim-license.txt) for information on Isaac Sim licensing.

Note that the `isaaclab_mimic` extension requires cuRobo, which has proprietary licensing terms that can be found in [`docs/licenses/dependencies/cuRobo-license.txt`](docs/licenses/dependencies/cuRobo-license.txt).

## Acknowledgement

Isaac Lab development initiated from the [Orbit](https://isaac-orbit.github.io/) framework. We would appreciate if
you would cite it in academic publications as well:

```
@article{mittal2023orbit,
   author={Mittal, Mayank and Yu, Calvin and Yu, Qinxi and Liu, Jingzhou and Rudin, Nikita and Hoeller, David and Yuan, Jia Lin and Singh, Ritvik and Guo, Yunrong and Mazhar, Hammad and Mandlekar, Ajay and Babich, Buck and State, Gavriel and Hutter, Marco and Garg, Animesh},
   journal={IEEE Robotics and Automation Letters},
   title={Orbit: A Unified Simulation Framework for Interactive Robot Learning Environments},
   year={2023},
   volume={8},
   number={6},
   pages={3740-3747},
   doi={10.1109/LRA.2023.3270034}
}
```
