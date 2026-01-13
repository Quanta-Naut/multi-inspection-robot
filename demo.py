"""
DDPG Demo Script
================
Runs trained DDPG agent in the Multi-Robot Environment.
Controls all robots using the same shared policy.

Usage:
    python demo.py --model models/ddpg_final
    python demo.py --dummy     # Static visualization
"""

import argparse
import time
import os
import pygame
import numpy as np
import torch

from multi_robot_env import MultiRobotEnv
from ddpg import DDPGAgent

def run_demo(args):
    # Initialize Environment
    env = MultiRobotEnv(
        num_robots=args.num_robots,
        num_humans=8,
        num_targets=4,
        wall_layout="complex" if args.complex else "simple",
        render_mode="human"
    )
    
    # Disable auto-movement for secondary robots (we control them manually via policy)
    env.disable_secondary_auto = True
    
    print("\n" + "=" * 60)
    if args.dummy:
        print("🛡️  DUMMY MODE (Static Visualization)")
        print("   Elements placed randomly. No movement.")
    else:
        print("🤖 DDPG AGENT DEMO")
        print(f"   Model: {args.model}")
    print("=" * 60)
    
    # Load Agent (if not dummy)
    agent = None
    if not args.dummy:
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        max_action = float(env.action_space.high[0])
        
        agent = DDPGAgent(state_dim, action_dim, max_action)
        try:
            agent.load(args.model)
            print("✅ Model loaded successfully")
        except Exception as e:
            print(f"⚠️  Error loading model: {e}")
            print("   Running with UNTRAINED (Random) agent")
            agent = None # Fallback to random
    
    # Main Loop
    for episode in range(args.episodes):
        obs, _ = env.reset()
        done = False
        
        while not done:
            # Render
            env.render()
            
            # Event Handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    return
                if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                    env.close()
                    return
            
            # Dummy Mode Loop (Static)
            if args.dummy:
                time.sleep(0.1)
                continue
            
            # Policy Control
            # We need to get actions for ALL robots
            # The env.step() takes action for Robot 0
            # We manually update other robots
            
            # 1. Get action for Primary Robot (Robot 0)
            # obs is already for Robot 0
            if agent:
                action_0 = agent.select_action(obs)
            else:
                action_0 = env.action_space.sample()
            
            # 2. Get actions for Secondary Robots and apply manually
            # Because env.step only controls Robot 0 by default
            for i in range(1, env.num_robots):
                robot_obs = env.get_observation(i)
                if agent:
                    action_i = agent.select_action(robot_obs)
                else:
                    action_i = env.action_space.sample()
                
                # Apply movement manually for secondary robots
                # We reuse the internal _move_robot logic or just simplified math
                # To be consistent, we should use the environment's physics
                # But _move_robot calculates rewards. We just want to move.
                # Let's reproduce the movement physics briefly
                
                robot = env.robots[i]
                linear_vel = float(action_i[0]) * 2.0 # MAX_LINEAR_VELOCITY import unavailable, hardcoded 2.0
                angular_vel = float(action_i[1]) * (np.pi/4)
                
                robot.heading = (robot.heading + angular_vel) % (2*np.pi)
                new_x = robot.x + linear_vel * np.cos(robot.heading)
                new_y = robot.y + linear_vel * np.sin(robot.heading)
                
                if env._is_valid_position(new_x, new_y, 2.0): # ROBOT_RADIUS
                    robot.x = new_x
                    robot.y = new_y
                    
                # Store path
                robot.path_history.append((robot.x, robot.y))
                if len(robot.path_history) > 200: robot.path_history.pop(0)

            # 3. Step Environment (moves Robot 0 and Humans)
            obs, reward, terminated, truncated, info = env.step(action_0)
            
            if terminated or truncated:
                done = True
                print(f"Episode Finished. Reward: {info['total_reward']:.2f}")
            
            time.sleep(args.speed)
            
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/ddpg_final", type=str)
    parser.add_argument("--episodes", default=5, type=int)
    parser.add_argument("--num_robots", default=2, type=int)
    parser.add_argument("--speed", default=0.05, type=float)
    parser.add_argument("--complex", action="store_true", help="Use complex wall layout")
    parser.add_argument("--dummy", action="store_true", help="Static dummy mode")
    
    args = parser.parse_args()
    run_demo(args)
