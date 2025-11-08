# Drone Navigation & Detection — RL + A* Notebook

A hands‑on Jupyter Notebook that builds a **custom grid‑world drone navigation environment**, integrates a **classical A* pathfinding** pipeline, and trains a **reinforcement learning (PPO)** agent to reach a target while avoiding obstacles. It also includes multiple **visualization utilities** (grid, path overlays, heatmaps, 3D cost surface, and a simple navigation graph using NetworkX).

> **What you’ll learn / get**
- Custom `gym.Env` environment with **8‑direction actions**, **2‑D position observations**, bounds checking, reward shaping, and terminal conditions.
- **A\*** pathfinding (Euclidean heuristic) for shortest‑path planning on a discrete grid.
- **PPO** policy gradient training via `stable-baselines3` to learn obstacle‑aware navigation.
- Visual tools: **grid renderers, A\* path overlay, dynamic step‑by‑step frames, heatmaps of traversal costs, and a 3D surface plot** of path costs.
- A simple **navigation graph** demo with edge weights (NetworkX).

---

##   Repository Structure

```
.
├── Drone_Navigation_Detection (1).ipynb   # The main notebook
└── README.md                              # You are here
```

---

##   Quick Start

### 1) Clone & enter
```bash
git clone <your-repo-url>.git
cd <your-repo-folder>
```

### 2) Create environment (recommended)
```bash
python -m venv .venv
# activate: Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 3) Install dependencies
> Newer versions of Stable‑Baselines3 use **gymnasium** internally.  
> This notebook imports `gym`. The easiest way is to install both `gym` and `gymnasium` and let SB3 use gymnasium under the hood.

```bash
pip install --upgrade pip
pip install "numpy" "matplotlib" "networkx"
pip install "gym==0.26.*" "gymnasium==0.29.*"
pip install "stable-baselines3[extra]"
```

If you hit import issues on some platforms, try:
```bash
pip install "stable-baselines3[extra]==2.3.0"
```

### 4) Run the notebook
```bash
jupyter notebook "Drone_Navigation_Detection (1).ipynb"
```
Execute the cells **top‑to‑bottom**.

---

##   Features

### Custom Drone Environment (`gym.Env`)
- **Actions**: 8 directions (N, S, E, W + diagonals).
- **Observation**: `(x, y)` position on a bounded 2‑D grid.
- **Rewards**: step penalty (−1), obstacle collision penalty (−10), goal reward (+10).
- **Terminations**: reaching the target or colliding with an obstacle.
- **Exploration**: simple ε‑greedy (`epsilon = 0.1`) used in the `step` logic.

### A* Pathfinding
- **Heuristic**: Euclidean distance.
- **Grid constraints**: bounds checks + obstacle avoidance.
- **Output**: optimal path as a sequence of lattice coordinates.

### PPO Reinforcement Learning
- **Library**: `stable-baselines3`
- **Policy**: `MlpPolicy`
- **Usage**:
  ```python
  from stable_baselines3 import PPO

  env = DroneEnv()
  model = PPO("MlpPolicy", env, verbose=1)
  model.learn(total_timesteps=10_000)
  ```
- **Testing**
  ```python
  obs = env.reset()
  for _ in range(20):
      action, _ = model.predict(obs)
      obs, reward, done, info = env.step(action)
      if done:
          break
  ```

### Visualizations
- **Environment grid** with obstacles and goal overlay.
- **A\* path overlay** from start to goal.
- **Dynamic frames** showing the drone moving across steps (good for demos).
- **Heatmap** of A\* traversal (g‑scores) to see low‑ vs high‑cost regions.
- **3D surface** plot of pathfinding costs.
- **Navigation graph** (NetworkX) with weighted edges.

---

##   Dependencies

- Python 3.9+
- `numpy`, `matplotlib`, `networkx`
- `gym` (0.26.x) and/or `gymnasium` (0.29.x)
- `stable-baselines3[extra]`

> If you only have `gymnasium`, you can adapt the environment signature slightly (e.g., returning `(obs, reward, terminated, truncated, info)` in `step`). The provided notebook uses classic `gym`’s `(obs, reward, done, info)`.

---

##   How It Works (High Level)

1. **Environment**: The drone starts at a fixed `(x, y)`, must reach a `target` while avoiding `obstacles`. Each move costs −1; the agent is rewarded for reaching the goal and penalized for collisions.
2. **A\***: Provides a **deterministic** baseline shortest path on the grid. This is useful for benchmarking and visual comparison against the RL agent’s behavior.
3. **RL (PPO)**: Learns a **stochastic policy** that maximizes expected return via policy gradients. Over time, the agent discovers paths that minimize penalties and reach the goal efficiently.
4. **Visualization**: Multiple views help understand why the agent or A\* makes certain choices (e.g., high cost regions around dense obstacle clusters).

---

##   Reproducible Demo

- **A\*** demo (example):
  ```python
  obstacles = [(6, 6), (7, 7)]
  path = a_star(start=(5, 5), goal=(8, 8), obstacles=obstacles, grid_width=11, grid_height=11)
  print("A* path:", path)
  ```

- **RL** demo (example):
  ```python
  env = DroneEnv()
  model = PPO("MlpPolicy", env, verbose=1)
  model.learn(total_timesteps=10_000)
  ```

- **Visuals**:
  ```python
  visualize_environment(drone_pos=(5, 5), target_pos=(8, 8), obstacles=obstacles, grid_size=(11,11))
  visualize_astar_path(drone_pos=(5, 5), target_pos=(8, 8), obstacles=obstacles, path=path, grid_size=(11,11))
  visualize_cost_heatmap(start=(5,5), goal=(8,8), obstacles=obstacles, grid_width=11, grid_height=11)
  visualize_cost_surface_3d(start=(5,5), goal=(8,8), obstacles=obstacles, grid_width=11, grid_height=11)
  visualize_navigation_graph()
  ```

---

##   Results & What to Expect

- **A\*** quickly returns an optimal grid path if one exists.
- **PPO** should steadily improve performance within ~10k+ steps on this small grid.
- Visualizations will help confirm the **shortest path region** (via heatmap/3D surface) and whether the agent is learning to **skirt obstacles**.

> Tip: Increase `total_timesteps` (e.g., 100k) for more stable policies. Tune rewards and obstacle layouts for richer behavior.

---

##  Extending the Project

- **Continuous control** with velocity/acceleration actions.
- **Sensor noise** and **partial observability**.
- **Curriculum learning** (progressively harder obstacle fields).
- Replace `gym` API with **gymnasium**’s `(terminated, truncated)` signals for modern stacks.
- Integrate **collision buffers**, **no‑fly zones**, or **wind fields**.

---

##   License

MIT — free to use and modify.

---

##   Acknowledgements
This project was developed at **Maulana Azad National Institute of Technology (MANIT), Bhopal**,  
under the guidance of **Dr. Dhirendra Pratap Singh**. 

- [`stable-baselines3`](https://github.com/DLR-RM/stable-baselines3)
- [`gym` / `gymnasium`](https://www.gymlibrary.dev/)
- [`networkx`](https://networkx.org/)
- The open‑source Python community ❤️
