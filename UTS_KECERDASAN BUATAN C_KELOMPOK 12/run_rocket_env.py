import pygame
import numpy as np
from stable_baselines3 import DQN
from rocket_env import SimpleRocketEnv  

# Fungsi Simulasi Rollout (Prediksi Lintasan)
def simulate_rollout(model, state, n_steps=400, deterministic=False):
    env_sim = SimpleRocketEnv()
    env_sim.set_state(state)

    traj = []
    obs = env_sim._get_obs()

    for _ in range(n_steps):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env_sim.step(int(action))
        traj.append((env_sim.state[0], env_sim.state[1]))
        if terminated or truncated:
            break

    return env_sim.state[0], env_sim.state[1], traj


# Konversi Koordinat Dunia ke Layar Pygame
def world_to_screen(x, y, screen_w, screen_h, scale=20):
    sx = int(screen_w // 2 + x * scale)
    sy = int(screen_h - y * scale)
    return sx, sy

# Program utama
if __name__ == "__main__":
    model = DQN.load("./models/dqn_baseline.zip")

    env = SimpleRocketEnv(render_mode="human")
    obs, _ = env.reset()
    done = False
    total_reward = 0
    best_reward = -9999
    landed_successfully = False
    landed_time = None  # waktu saat landing

    pygame.init()
    screen_w, screen_h = 960, 540
    screen = pygame.display.set_mode((screen_w, screen_h))
    pygame.display.set_caption("Rocket Landing Prediction (AI)")

    clock = pygame.time.Clock()

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        # Kalau belum landed → mainkan environment & update reward
        if not landed_successfully:
            state = env._get_obs()

            preds = []
            for _ in range(5):
                fx, fy, traj = simulate_rollout(model, state, n_steps=400, deterministic=False)
                preds.append((fx, fy))

            avg_x = np.mean([p[0] for p in preds])
            avg_y = np.mean([p[1] for p in preds])

            # Langkah DQN
            action, _ = model.predict(state, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))

            total_reward += reward
            best_reward = max(best_reward, total_reward)

            # Jika mendarat → stop update dan catat waktu
            if terminated or truncated:
                landed_successfully = True
                landed_time = pygame.time.get_ticks()

        # ---- Render visual ----
        screen.fill((0, 0, 0))

        # Platform
        platform_x = env.target_pos[0]
        platform_y = env.target_pos[1]
        px1, py1 = world_to_screen(platform_x - 3.0, platform_y, screen_w, screen_h)
        px2, py2 = world_to_screen(platform_x + 3.0, platform_y + 0.5, screen_w, screen_h)
        pygame.draw.rect(screen, (0, 100, 255), pygame.Rect(px1, py2, px2 - px1, py1 - py2))

        # Roket
        rx, ry = world_to_screen(env.state[0], env.state[1], screen_w, screen_h)
        pygame.draw.circle(screen, (0, 255, 0), (rx, ry), 6)

        # Lintasan Prediksi Hanya Muncul Sebelum Landing
        if not landed_successfully:
            for fx, fy in preds:
                sx, sy = world_to_screen(fx, fy, screen_w, screen_h)
                pygame.draw.circle(screen, (255, 0, 0), (sx, sy), 5)

            ax, ay = world_to_screen(avg_x, avg_y, screen_w, screen_h)
            pygame.draw.circle(screen, (255, 255, 0), (ax, ay), 7)

        # Teks Reward
        font = pygame.font.Font(None, 28)
        text_reward = font.render(f"Reward: {total_reward:.2f}", True, (255, 255, 255))
        text_best = font.render(f"Best: {best_reward:.2f}", True, (255, 255, 0))
        screen.blit(text_reward, (20, 20))
        screen.blit(text_best, (20, 50))

        # tampilkan pesan landing
        if landed_successfully:
            font_big = pygame.font.Font(None, 48)
            msg = font_big.render("Landed Successfully!", True, (0, 150, 255))
            screen.blit(msg, (screen_w // 2 - 200, screen_h // 2 - 50))

            # tutup otomatis 3 detik setelah landing
            if landed_time and pygame.time.get_ticks() - landed_time > 3000:
                done = True

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()

