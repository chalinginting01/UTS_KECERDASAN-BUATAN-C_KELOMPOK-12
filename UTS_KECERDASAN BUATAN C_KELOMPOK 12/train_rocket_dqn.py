
import os
import math
import argparse
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt
import torch as th
import torch.nn as nn
from stable_baselines3.dqn.policies import DQNPolicy

USE_PER_AVAILABLE = False
PrioritizedReplayBuffer = None
try:
    # sb3_contrib location can vary by version; try common locations
    # If not installed, this import will fail and we'll fall back to vanilla replay buffer.
    from sb3_contrib.common.buffers import PrioritizedReplayBuffer  # type: ignore
    USE_PER_AVAILABLE = True
except Exception:
    try:
        # alternative path (older/newer sb3_contrib layouts)
        from sb3_contrib.replay_buffers import PrioritizedReplayBuffer  # type: ignore
        USE_PER_AVAILABLE = True
    except Exception:
        USE_PER_AVAILABLE = False
        PrioritizedReplayBuffer = None

# ============================================================
# 1. ENVIRONMENT : SimpleRocketEnv
# ============================================================
class SimpleRocketEnv(gym.Env):
    
    metadata = {"render.modes": ["human"]}

    def __init__(self, render_mode=None, dt=0.05):
        super().__init__()
        self.dt = float(dt)
        # world limits (meters)
        self.x_limit = 10.0
        self.y_limit = 12.0

        # dynamics params
        self.mass = 1.0
        self.main_thrust = 15.0
        self.side_thrust = 4.0
        self.torque = 3.0
        self.gravity = 9.81
        self.drag = 0.1

        # action / observation
        self.action_space = spaces.Discrete(4)
        high = np.array(
            [self.x_limit, self.y_limit, 50.0, 50.0, 1.0, 1.0, 50.0, self.x_limit, self.y_limit],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        # landing target / thresholds
        self.target_y = 1.0
        self.target_speed = 0.5
        self.land_dist_thresh = 0.8
        self.land_angle_thresh = 0.3
        self.land_speed_thresh = 1.5
        self.max_steps = 800

        # rendering
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.font = None
        self.rocket_surface = None
        self.target_surface = None

        # moving target initial (set in reset)
        self.target = {"x": 0.0, "y": self.target_y, "vx": self.target_speed}

        # seed & state
        self.seed()
        self.state = None
        self.step_count = 0
        self._terminated = False

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        x = self.np_random.uniform(-3.0, 3.0)
        y = self.np_random.uniform(6.0, 10.0)
        vx = self.np_random.normal(0.0, 1.0)
        vy = self.np_random.normal(-1.0, 1.0)
        theta = self.np_random.normal(0.0, 0.15)
        omega = self.np_random.normal(0.0, 0.4)
        target_x = self.np_random.uniform(-4.0, 4.0)
        self.target = {"x": float(target_x), "y": float(self.target_y), "vx": float(self.target_speed)}
        self.state = np.array(
            [x, y, vx, vy, math.sin(theta), math.cos(theta), omega, self.target["x"] - x, self.target["y"] - y],
            dtype=np.float32,
        )
        self.step_count = 0
        self._terminated = False
        return self._get_obs(), {}

    def _get_obs(self):
        return self.state.copy()

    def step(self, action):
      
        if self._terminated:
            # If environment already terminated, return zero reward and done flags.
            return self._get_obs(), 0.0, True, True, {}

        x, y, vx, vy, sint, cost, omega, dx, dy = self.state
        theta = math.atan2(float(sint), float(cost))

        # --- target motion (simple bounce inside x limits) ---
        self.target["x"] += self.target["vx"] * self.dt
        if self.target["x"] > self.x_limit - 0.5 or self.target["x"] < -self.x_limit + 0.5:
            self.target["vx"] *= -1.0

        # --- decode action into forces/torque ---
        thrust_x = 0.0
        thrust_y = 0.0
        torque = 0.0
        if action == 1:  # main thruster (aligned with body)
            thrust_y += self.main_thrust * math.cos(theta)
            thrust_x += self.main_thrust * math.sin(theta)
        elif action == 2:  # left torque + side thrust
            torque += self.torque
            thrust_x -= self.side_thrust
        elif action == 3:  # right torque + side thrust
            torque -= self.torque
            thrust_x += self.side_thrust
        # action == 0 -> idle

        # --- dynamics integration (semi-implicit Euler) ---
        ax = (thrust_x - self.drag * vx) / self.mass
        ay = (thrust_y - self.drag * vy - self.mass * self.gravity) / self.mass
        vx = vx + ax * self.dt
        vy = vy + ay * self.dt
        x = x + vx * self.dt
        y = y + vy * self.dt
        domega = torque / (self.mass * 0.5)
        omega = omega + domega * self.dt
        theta = theta + omega * self.dt

        # --- compute distances to target ---
        dx = self.target["x"] - x
        dy = self.target["y"] - y

        # clamp positions to world limits
        x = float(np.clip(x, -self.x_limit * 2.0, self.x_limit * 2.0))
        y = float(np.clip(y, 0.0, self.y_limit * 2.0))

        # update state
        self.state = np.array([x, y, vx, vy, math.sin(theta), math.cos(theta), omega, dx, dy], dtype=np.float32)

        self.step_count += 1
        terminated = False
        truncated = False
        info = {}

        # --- termination conditions ---
        if abs(x) > self.x_limit * 1.5 or y > self.y_limit * 1.5 or y <= 0.0:
            terminated = True
            success = False
        else:
            dist = math.hypot(dx, dy)
            angle = abs(theta)
            speed = math.hypot(vx, vy)
            if dist <= self.land_dist_thresh and angle <= self.land_angle_thresh and speed <= self.land_speed_thresh:
                terminated = True
                success = True
            else:
                success = False

        # --- reward shaping ---
        reward = 0.0
        reward -= 0.5 * math.hypot(dx, dy)          # penalize distance to target
        reward -= 0.2 * math.hypot(vx, vy)          # penalize speed
        reward += 1.5 * math.cos(theta)             # upright bonus
        reward -= 0.05                              # time penalty
        if abs(dx) < 1.5 and vy < 0:
            reward += 1.0

        if terminated and success:
            reward += 250.0
            info["success"] = True
            self._terminated = True  # stop further simulation & give final reward only once
        elif terminated and not success:
            reward -= 100.0
            info["success"] = False
            self._terminated = True

        if self.step_count >= self.max_steps:
            truncated = True
            self._terminated = True

        return self._get_obs(), float(reward), terminated, truncated, info

    # simple rendering using pygame if desired
    def render(self, mode="human"):
        if mode != "human":
            return
        try:
            import pygame  # local import in case headless
        except Exception:
            return

        if self.screen is None:
            pygame.init()
            # Map world coords to screen size
            self.screen_w = 640
            self.screen_h = 480
            self.screen = pygame.display.set_mode((self.screen_w, self.screen_h))
            pygame.display.set_caption("SimpleRocketEnv")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("Arial", 14)
            # create simple surfaces
            self.rocket_surface = pygame.Surface((30, 60), pygame.SRCALPHA)
            pygame.draw.polygon(self.rocket_surface, (220, 220, 220), [(15, 0), (30, 60), (0, 60)])
            self.target_surface = pygame.Surface((int(2.0 * 50), 16), pygame.SRCALPHA)
            pygame.draw.rect(self.target_surface, (30, 120, 220), (0, 0, self.target_surface.get_width(), 16))

        # draw background
        self.screen.fill((0, 0, 20))

        # convert world -> screen
        def world_to_screen(px, py):
            sx = int(self.screen_w // 2 + px * 20.0)
            sy = int(self.screen_h - py * 20.0)
            return sx, sy

        x, y, vx, vy, sint, cost, omega, dx, dy = self.state
        sx, sy = world_to_screen(x, y)
        tx, ty = world_to_screen(self.target["x"], self.target["y"])

        # draw rocket (no rotation for simplicity)
        rect = self.rocket_surface.get_rect(center=(sx, sy))
        self.screen.blit(self.rocket_surface, rect.topleft)

        # draw target
        trect = self.target_surface.get_rect(center=(tx, ty))
        self.screen.blit(self.target_surface, trect.topleft)

        info = f"x={x:.2f} y={y:.2f} vx={vx:.2f} vy={vy:.2f}"
        self.screen.blit(self.font.render(info, True, (255, 255, 255)), (8, 8))

        if self._terminated:
            msg = self.font.render("LANDING COMPLETED (reward stopped)", True, (0, 255, 0))
            self.screen.blit(msg, (8, 28))

        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        if self.screen is not None:
            import pygame
            pygame.quit()
            self.screen = None


# ============================================================
# 2. Dueling DQN Policy 
# ============================================================
class DuelingQNetwork(nn.Module):
    def __init__(self, input_dim, action_dim):
        super().__init__()
        self.feature = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())
        self.value_stream = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 1))
        self.adv_stream = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, action_dim))

    def forward(self, x):
        x = self.feature(x)
        value = self.value_stream(x)
        adv = self.adv_stream(x)
        # combine value & advantage -> Q(s,a)
        return value + adv - adv.mean(dim=1, keepdim=True)


class DuelingDQNPolicy(DQNPolicy):
    def _build_q_net(self) -> None:
        # features_dim is inferred by DQNPolicy; build custom networks
        self.q_net = DuelingQNetwork(self.features_dim, self.action_space.n)
        self.q_net_target = DuelingQNetwork(self.features_dim, self.action_space.n)
        # copy weights
        self.q_net_target.load_state_dict(self.q_net.state_dict())


# ============================================================
# 3. Helper functions
# ============================================================
def make_env(seed: int = 0):
    def _init():
        env = SimpleRocketEnv()
        env.seed(seed)
        return env
    return DummyVecEnv([_init])


def plot_rewards(rewards, model_dir, title):
    os.makedirs(model_dir, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.plot(rewards, marker="o")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(title)
    plt.grid(True)
    plt.savefig(os.path.join(model_dir, "learning_curve.png"), dpi=150)
    plt.close()


# ============================================================
# 4. TRAINING FUNCTIONS
# ============================================================
def train_dqn(total_timesteps=500_000, model_dir="./models_dqn", use_per=False):
    os.makedirs(model_dir, exist_ok=True)
    env = make_env()
    print("ENV created; use_per =", use_per, " PER available =", USE_PER_AVAILABLE)

    replay_buffer_class = None
    replay_buffer_kwargs = {}
    if use_per:
        if USE_PER_AVAILABLE and PrioritizedReplayBuffer is not None:
            replay_buffer_class = PrioritizedReplayBuffer
            # example kwargs - actual constructor args depend on sb3_contrib version
            # stable-baselines3 DQN accepts replay_buffer_class and replay_buffer_kwargs
            replay_buffer_kwargs = {"alpha": 0.6, "beta": 0.4}
            print("Using PrioritizedReplayBuffer from sb3_contrib")
        else:
            print("sb3_contrib not installed or PrioritizedReplayBuffer not found; falling back to standard replay buffer.")
            replay_buffer_class = None

    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        buffer_size=100_000,
        learning_starts=5_000,
        batch_size=128,
        learning_rate=5e-4,
        target_update_interval=1000,
        train_freq=4,
        exploration_fraction=0.3,
        exploration_final_eps=0.02,
        tensorboard_log="./tb_logs_dqn/",
        replay_buffer_class=replay_buffer_class,
        replay_buffer_kwargs=replay_buffer_kwargs,
    )
    model.learn(total_timesteps=total_timesteps, tb_log_name="DQN_Baseline")
    print("🚀 Training DQN baseline ...")
    model.learn(total_timesteps=total_timesteps)
    model.save(os.path.join(model_dir, "dqn_baseline.zip"))
    print("✅ DQN training complete.")
    return model


def train_dueling(total_timesteps=500_000, model_dir="./models_dueling", use_per=False):
    os.makedirs(model_dir, exist_ok=True)
    env = make_env()
    replay_buffer_class = None
    replay_buffer_kwargs = {}
    if use_per and USE_PER_AVAILABLE and PrioritizedReplayBuffer is not None:
        replay_buffer_class = PrioritizedReplayBuffer
        replay_buffer_kwargs = {"alpha": 0.6, "beta": 0.4}
        print("Using PrioritizedReplayBuffer for dueling model")
    model = DQN(
        DuelingDQNPolicy,
        env,
        verbose=1,
        buffer_size=100_000,
        learning_starts=5_000,
        batch_size=128,
        learning_rate=5e-4,
        target_update_interval=1000,
        train_freq=4,
        exploration_fraction=0.3,
        exploration_final_eps=0.02,
        tensorboard_log="./tb_logs_dueling/",
        replay_buffer_class=replay_buffer_class,
        replay_buffer_kwargs=replay_buffer_kwargs,
    )
    model.learn(total_timesteps=total_timesteps, tb_log_name="Dueling_DQN")
    print("🚀 Training Dueling DQN ...")
    model.learn(total_timesteps=total_timesteps)
    model.save(os.path.join(model_dir, "dueling_dqn.zip"))
    print("✅ Dueling DQN training complete.")
    return model


def train_double(total_timesteps=500_000, model_dir="./models_double", use_per=False):
    # SB3's DQN is Double-DQN style by default; provide tuned settings
    os.makedirs(model_dir, exist_ok=True)
    env = make_env()
    replay_buffer_class = None
    replay_buffer_kwargs = {}
    if use_per and USE_PER_AVAILABLE and PrioritizedReplayBuffer is not None:
        replay_buffer_class = PrioritizedReplayBuffer
        replay_buffer_kwargs = {"alpha": 0.6, "beta": 0.4}
    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        buffer_size=100_000,
        learning_starts=5_000,
        batch_size=128,
        learning_rate=5e-4,
        target_update_interval=1000,
        train_freq=4,
        exploration_fraction=0.3,
        exploration_final_eps=0.02,
        tensorboard_log="./tb_logs_double/",
        replay_buffer_class=replay_buffer_class,
        replay_buffer_kwargs=replay_buffer_kwargs,
    )
    model.learn(total_timesteps=total_timesteps, tb_log_name="Double_DQN")
    print("🚀 Training Double DQN (SB3 default/Double logic) ...")
    model.learn(total_timesteps=total_timesteps)
    model.save(os.path.join(model_dir, "double_dqn.zip"))
    print("✅ Double DQN training complete.")
    return model


# ============================================================
# 5. MAIN (CLI)
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Train DQN variants on SimpleRocketEnv")
    parser.add_argument("--timesteps", type=int, default=500_000, help="Total training timesteps per model")
    parser.add_argument("--dueling", action="store_true", help="Train dueling DQN (in addition to baseline)")
    parser.add_argument("--double", action="store_true", help="Train double DQN (in addition to baseline)")
    parser.add_argument("--per", action="store_true", help="Use Prioritized Replay Buffer (requires sb3_contrib)")
    args = parser.parse_args()

    total_timesteps = args.timesteps
    print("=== Starting Rocket Training Suite ===")
    print(f"Timesteps per model: {total_timesteps}")
    # baseline
    dqn_model = train_dqn(total_timesteps=total_timesteps, model_dir="./models_dqn", use_per=args.per)
    # dueling
    if args.dueling:
        dueling_model = train_dueling(total_timesteps=total_timesteps, model_dir="./models_dueling", use_per=args.per)
    # double
    if args.double:
        double_model = train_double(total_timesteps=total_timesteps, model_dir="./models_double", use_per=args.per)

    print("✅ All requested training complete.")


if __name__ == "__main__":
    main()

