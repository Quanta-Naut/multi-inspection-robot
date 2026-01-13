"""
Training Script for DDPG
========================
Trains a DDPG agent to navigate the Multi-Robot Environment.

Usage:
    python train.py --episodes 500 --batch_size 64
"""

import os
import argparse
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from multi_robot_env import MultiRobotEnv
from ddpg import DDPGAgent, ReplayBuffer


def train(args):
    # Create environment
    env = MultiRobotEnv(
        grid_size=100,
        num_robots=1,   # Training single robot efficiency first is usually better
        num_humans=args.num_humans,
        num_targets=3
    )
    
    # Set seeds
    env.reset(seed=args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # State and Action dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] 
    max_action = float(env.action_space.high[0])
    
    # Initialize Agent
    agent = DDPGAgent(state_dim, action_dim, max_action)
    
    # Replay Buffer
    replay_buffer = ReplayBuffer(state_dim, action_dim, max_size=100000)
    
    # Create model directory
    if not os.path.exists("models"):
        os.makedirs("models")
    
    # Metrics
    episode_rewards = []
    
    print(f"---------------------------------------")
    print(f"Policy: DDPG, Env: Multi-Robot, Seed: {args.seed}")
    print(f"---------------------------------------")

    # Training Loop
    total_steps = 0
    
    for episode in range(args.episodes):
        obs, _ = env.reset()
        episode_reward = 0
        terminated = False
        truncated = False
        step = 0
        
        while not (terminated or truncated) and step < env.max_steps:
            step += 1
            total_steps += 1
            
            # Select action with exploration noise
            if total_steps < args.start_steps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(obs, noise=args.expl_noise)
            
            # Perform action
            new_obs, reward, terminated, truncated, info = env.step(action)
            
            # Store in buffer
            done_bool = float(terminated or truncated) # DDPG uses done flag
            replay_buffer.add(obs, action, new_obs, reward, done_bool)
            
            obs = new_obs
            episode_reward += reward
            
            # Train agent
            if len(replay_buffer.state) > args.batch_size:
                agent.train(replay_buffer, args.batch_size)
        
        # Logging
        episode_rewards.append(episode_reward)
        avg_reward = np.mean(episode_rewards[-20:])
        
        print(f"Episode {episode+1}: Reward {episode_reward:.2f} | Avg (20): {avg_reward:.2f} | Steps: {step}")
        
        # Save model
        if (episode + 1) % 50 == 0:
            agent.save(f"models/ddpg_model_{episode+1}")
            print(f"Saved model at episode {episode+1}")
            
    # Save final model
    agent.save("models/ddpg_final")
    print("Training Complete. Saved final model.")
    
    # Plot training curve
    plt.figure()
    plt.plot(episode_rewards)
    plt.title("DDPG Training Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.savefig("models/training_curve.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", default=500, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--start_steps", default=1000, type=int, help="Steps before using policy (random exploration)")
    parser.add_argument("--expl_noise", default=0.1, type=float, help="Std of Gaussian exploration noise")
    parser.add_argument("--num_humans", default=6, type=int)
    args = parser.parse_args()
    
    train(args)
