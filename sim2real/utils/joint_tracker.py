import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Any
import csv
import gymnasium as gym

# Number of timesteps to show in plots (last N steps)
PLOT_LAST_N_TIMESTEPS = 100


class JointTrackingLogger:
    def __init__(
        self, num_joints: int, joint_names: Optional[List[str]] = None, dt: float = 0.02
    ):
        self.num_joints = num_joints
        self.dt = dt

        if joint_names is None:
            self.joint_names = [f"Joint {i}" for i in range(num_joints)]
        else:
            self.joint_names = joint_names

        self.commanded_positions = []
        self.actual_positions = []
        self.timestamps = []
        self.current_time = 0.0

    def log_step(self, commanded_pos: np.ndarray, actual_pos: np.ndarray):
        self.commanded_positions.append(np.array(commanded_pos))
        self.actual_positions.append(np.array(actual_pos))
        self.timestamps.append(self.current_time)
        self.current_time += self.dt

    def reset(self):
        self.commanded_positions = []
        self.actual_positions = []
        self.timestamps = []
        self.current_time = 0.0

    def save_csv(self, filepath: str):
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)

            header = ["Time (s)"]
            for name in self.joint_names:
                header.extend([f"{name}_commanded", f"{name}_actual"])
            writer.writerow(header)

            for i, t in enumerate(self.timestamps):
                row = [t]
                for j in range(self.num_joints):
                    row.extend(
                        [self.commanded_positions[i][j], self.actual_positions[i][j]]
                    )
                writer.writerow(row)

    def plot(self, save_path: Optional[str] = None, show: bool = False):
        if not self.timestamps:
            print("No data to plot")
            return

        commanded = np.array(self.commanded_positions)
        actual = np.array(self.actual_positions)
        times = np.array(self.timestamps)

        if len(times) > PLOT_LAST_N_TIMESTEPS:
            commanded = commanded[-PLOT_LAST_N_TIMESTEPS:]
            actual = actual[-PLOT_LAST_N_TIMESTEPS:]
            times = times[-PLOT_LAST_N_TIMESTEPS:]

        n_cols = 3
        n_rows = int(np.ceil(self.num_joints / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)

        for i in range(self.num_joints):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col]

            ax.plot(
                times, commanded[:, i], "b--", label="Target", linewidth=1.5, alpha=0.7
            )
            ax.plot(
                times, actual[:, i], "orange", label="Actual", linewidth=1.5, alpha=0.9
            )
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Position (rad)")
            ax.set_title(self.joint_names[i])
            ax.legend(loc="upper right")
            ax.grid(True, alpha=0.3)

        for i in range(self.num_joints, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].axis("off")

        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Plot saved to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def get_tracking_error_stats(self) -> Dict[str, Any]:
        if not self.timestamps:
            return {}

        commanded = np.array(self.commanded_positions)
        actual = np.array(self.actual_positions)
        errors = actual - commanded

        stats = {}
        for i in range(self.num_joints):
            joint_error = errors[:, i]
            stats[self.joint_names[i]] = {
                "mean_error": float(np.mean(joint_error)),
                "std_error": float(np.std(joint_error)),
                "max_error": float(np.max(np.abs(joint_error))),
                "rmse": float(np.sqrt(np.mean(joint_error**2))),
            }

        return stats


class JointTrackingWrapper(gym.Wrapper):
    def __init__(self, env, dt: float = 0.02):
        super().__init__(env)

        joint_names = self._extract_joint_names()

        num_joints = env.action_space.shape[0]

        self.logger = JointTrackingLogger(
            num_joints=num_joints, joint_names=joint_names, dt=dt
        )

        self._last_action = None
        self._default_pose = None

        self._extract_env_config()

    def _extract_joint_names(self):
        try:
            if hasattr(self.env, "env") and hasattr(self.env.env, "mj_model"):
                model = self.env.env.mj_model
                joint_names = []
                for i in range(1, model.njnt):  # Skip first joint (free joint)
                    joint_name = model.joint(i).name
                    if joint_name:
                        joint_names.append(joint_name)
                return joint_names
        except Exception:
            pass
        return None

    def _extract_env_config(self):
        try:
            if hasattr(self.env, "env"):
                env_obj = self.env.env
                if hasattr(env_obj, "_default_pose"):
                    self._default_pose = np.array(env_obj._default_pose)
                if hasattr(env_obj, "_config") and hasattr(
                    env_obj._config, "action_scale"
                ):
                    self._action_scale = env_obj._config.action_scale
                else:
                    self._action_scale = 0.5  # Default from G1/Go2
        except Exception:
            self._action_scale = 0.5

    def reset(self, **kwargs):
        self.logger.reset()
        self._last_action = None
        return self.env.reset(**kwargs)

    def step(self, action):
        if isinstance(action, np.ndarray):
            self._last_action = action.copy()
        else:
            self._last_action = np.array(action)

        obs, reward, done, truncated, info = self.env.step(action)

        commanded_pos, actual_pos = self._extract_positions(action, info)

        if commanded_pos is not None and actual_pos is not None:
            self.logger.log_step(commanded_pos, actual_pos)

        return obs, reward, done, truncated, info

    def _extract_positions(self, action, info):
        commanded_pos = None
        actual_pos = None

        try:
            # Priority 1: From info dict
            if "motor_targets" in info:
                motor_targets = info["motor_targets"]
                if isinstance(motor_targets, np.ndarray):
                    if motor_targets.ndim > 1:
                        commanded_pos = motor_targets[0]
                    else:
                        commanded_pos = motor_targets
                else:
                    commanded_pos = np.array(motor_targets)

            # Priority 2: Calculate from action and default pose
            elif self._default_pose is not None:
                action_array = np.array(action)
                if action_array.ndim > 1:
                    action_array = action_array[0]
                commanded_pos = self._default_pose + action_array * self._action_scale

            # Priority 3: Just use action
            else:
                action_array = np.array(action)
                if action_array.ndim > 1:
                    commanded_pos = action_array[0]
                else:
                    commanded_pos = action_array

            # Get actual positions from environment state
            if hasattr(self.env, "env_state"):
                # JAX-based environment
                qpos = np.array(self.env.env_state.data.qpos)
                if qpos.ndim > 1:
                    actual_pos = qpos[0, 7:]  # Skip free joint (first 7 elements)
                else:
                    actual_pos = qpos[7:]
            elif hasattr(self.env, "data"):
                # MuJoCo data directly
                qpos = np.array(self.env.data.qpos)
                actual_pos = qpos[7:]

        except Exception as e:
            print(f"Warning: Could not extract positions: {e}")
            return None, None

        return commanded_pos, actual_pos

    def save_joint_tracking_plot(self, save_path: str):
        """Save joint tracking plot to file."""
        self.logger.plot(save_path=save_path, show=False)

    def save_joint_tracking_csv(self, save_path: str):
        """Save joint tracking data to CSV file."""
        self.logger.save_csv(save_path)

    def get_joint_tracking_stats(self) -> Dict[str, Any]:
        """Get joint tracking error statistics."""
        return self.logger.get_tracking_error_stats()


def create_tracked_env(
    env_name: str,
    device: str = "cuda",
    render_callback=None,
    command_type=None,
    dt: float = 0.02,
):
    import sys

    sys.path.append("..")
    from algorithms.utils.wrapper_gym import get_env

    env = get_env(env_name, device, render_callback, command_type)

    tracked_env = JointTrackingWrapper(env, dt=dt)

    return tracked_env
