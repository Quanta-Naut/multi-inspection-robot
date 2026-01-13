# Multi-Robot Navigation with DDPG

A clean, standalone implementation of **Deep Deterministic Policy Gradient (DDPG)** for multi-robot navigation in a dynamic environment.

## 🤖 Features

*   **Custom DDPG Implementation**: PyTorch-based Actor-Critic architecture.
*   **Dynamic Environment**: Robots must navigate to targets while avoiding:
    *   Dynamic crowds (humans with social zones)
    *   Static obstacles (walls)
*   **Social Awareness**: Reward function penalizes "social friction" (getting too close to humans).
*   **Multi-Agent Support**: Demo script allows one policy to control multiple robots.

## 📂 Structure

*   `ddpg.py`: Core algorithm (Actor, Critic, ReplayBuffer, DDPGAgent).
*   `multi_robot_env.py`: Custom Gymnasium environment.
*   `social_cost.py`: Helper for calculating social density/friction.
*   `train.py`: Main training loop.
*   `demo.py`: Visualization script.

## 🚀 Installation

1.  Requires Python 3.8+
2.  Install dependencies:
    ```bash
    pip install numpy torch gymnasium pygame matplotlib seaborn tqdm
    ```

## 🏃 Usage

### 1. Train the Agent
Train a new policy from scratch. Models are saved to `models/`.
```bash
python train.py --episodes 500
```
*Arguments:*
*   `--episodes`: Number of training episodes (default: 500)
*   `--num_humans`: Number of humans in environment (default: 6)
*   `--expl_noise`: Exploration noise std (default: 0.1)

### 2. Run the Demo
Watch the trained agents navigate.
```bash
# Run with trained model (final)
python demo.py --model models/ddpg_final

# Run with intermediate checkpoint
python demo.py --model models/ddpg_model_50

# Run with random initialization (if no model yet)
python demo.py --model none
```
*Arguments:*
*   `--num_robots`: Number of robots to spawn (default: 2)
*   `--dummy`: Run in static "dummy" mode to check environment/legend.
*   `--complex`: Use complex wall layout.

### 3. Dummy Mode (Visualization Check)
View the environment layout and legend without movement.
```bash
python demo.py --dummy
```

## 🧠 Algorithm Details

The agent uses **DDPG** (Deep Deterministic Policy Gradient), an off-policy algorithm for continuous action spaces.

*   **State Space (59 dim)**: Lidar (36), Human positions (16), Robot pose (3), Target (2), Crowd density (1), Battery (1).
*   **Action Space (2 dim)**: Linear velocity, Angular velocity.
*   **Reward Function**: 
    *   +100 for reaching target
    *   -50 for wall collision
    *   -5 for human collision
    *   -Social Friction cost
    *   +Forward movement bonus
    *   -Spinning penalty

---
*Created for Advanced Agentic Coding - Multi-Robot Navigation Project*
