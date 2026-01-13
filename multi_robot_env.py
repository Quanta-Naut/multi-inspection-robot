"""
Multi-Robot Navigation Environment
===================================
A custom Gymnasium environment for multi-robot navigation with
human obstacles, static walls, and social friction awareness.

Features:
- 2D grid world (100x100 cells)
- Multiple mobile robots with continuous actions
- Dynamic human obstacles with various movement patterns
- Static walls and obstacles
- Social friction-aware reward function
- Pygame visualization with Legend Overlay

Author: Multi-Robot RL Project
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import math

from social_cost import calculate_social_cost


# ============================================================================
# CONSTANTS & CONFIGURATION
# ============================================================================

# Environment
GRID_SIZE = 100
CELL_SIZE = 8  # Pixels per cell for rendering

# Physics
MAX_LINEAR_VELOCITY = 2.0
MAX_ANGULAR_VELOCITY = np.pi / 4  # 45 degrees per step
ROBOT_RADIUS = 2.0
HUMAN_RADIUS = 1.5
TARGET_RADIUS = 3.0

# Lidar
LIDAR_RAYS = 36  # Number of lidar rays (10° resolution)
LIDAR_MAX_RANGE = 20.0

# Rewards - FIXED to prevent spinning!
REWARD_TARGET_REACHED = 100.0      # Much higher to strongly incentivize reaching targets
REWARD_DISTANCE_PENALTY = -0.05    # Reduced - was making all actions equally bad
REWARD_HUMAN_COLLISION = -5.0
REWARD_WALL_COLLISION = -50.0      # Heavily penalized as requested
REWARD_HIGH_JERK = -0.5            # Reduced
REWARD_SOCIAL_FRICTION = -0.2      # Reduced
REWARD_TIME_PENALTY = -0.005       # Reduced
REWARD_FORWARD_MOVEMENT = 0.5      # NEW: Bonus for moving forward
REWARD_SPINNING_PENALTY = -1.0     # NEW: Penalty for high angular velocity

# Colors (for Pygame rendering)
COLORS = {
    'background': (26, 26, 46),      # Dark blue-gray
    'grid': (50, 50, 80),            # Subtle grid
    'wall': (100, 100, 120),         # Gray walls
    'robot': (78, 205, 196),         # Cyan
    'robot_direction': (255, 255, 255),
    'human': (255, 107, 107),        # Coral red
    'target': (149, 231, 122),       # Green
    'target_reached': (100, 180, 100),
    'lidar': (150, 150, 200, 100),   # Semi-transparent
    'path': (255, 200, 100),         # Orange trail
    'text': (255, 255, 255),
    'info_bg': (40, 40, 60),
}


# ============================================================================
# ENUMS AND DATA CLASSES
# ============================================================================

class HumanMovementPattern(Enum):
    """Movement patterns for human obstacles."""
    RANDOM_WALK = "random_walk"
    LINEAR_PATROL = "linear_patrol"
    GOAL_SEEKING = "goal_seeking"
    STATIONARY = "stationary"


@dataclass
class RobotState:
    """State of a single robot."""
    id: int
    x: float
    y: float
    heading: float = 0.0  # Radians
    velocity: float = 0.0
    battery: float = 100.0
    path_history: List[Tuple[float, float]] = field(default_factory=list)
    prev_velocity: float = 0.0
    collisions: int = 0
    wall_collisions: int = 0
    targets_reached: int = 0


@dataclass
class HumanState:
    """State of a human obstacle."""
    id: int
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    pattern: HumanMovementPattern = HumanMovementPattern.RANDOM_WALK
    patrol_points: List[Tuple[float, float]] = field(default_factory=list)
    current_patrol_idx: int = 0
    speed: float = 0.5


@dataclass
class TargetState:
    """State of an inspection target."""
    id: int
    x: float
    y: float
    reached: bool = False
    assigned_robot: Optional[int] = None


# ============================================================================
# ENVIRONMENT CLASS
# ============================================================================

class MultiRobotEnv(gym.Env):
    """
    Multi-Robot Navigation Environment with Social Awareness.
    
    A 2D grid world where robots must navigate to inspection targets
    while avoiding humans and maintaining social distance.
    
    Observation Space:
        - Robot pose: [x, y, heading] (3)
        - Lidar scan: distances at 36 angles (36)
        - Human positions: relative [x, y] for 8 humans (16)
        - Target location: relative [x, y] (2)
        - Crowd density: scalar (1)
        - Battery level: scalar (1)
        Total: 59 dimensions
        
    Action Space:
        - Linear velocity: [-1, 1] (scaled to max velocity)
        - Angular velocity: [-1, 1] (scaled to max angular velocity)
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(
        self,
        grid_size: int = GRID_SIZE,
        num_robots: int = 2,
        num_humans: int = 6,
        num_targets: int = 3,
        max_steps: int = 500,
        render_mode: Optional[str] = None,
        wall_layout: str = "simple"
    ):
        super().__init__()
        
        self.grid_size = grid_size
        self.num_robots = num_robots
        self.num_humans = num_humans
        self.num_targets = num_targets
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.wall_layout = wall_layout
        
        # Observation space
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(59,),
            dtype=np.float32
        )
        
        # Action space
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32
        )
        
        # Internal state
        self.robots: List[RobotState] = []
        self.humans: List[HumanState] = []
        self.targets: List[TargetState] = []
        self.walls: np.ndarray = np.zeros((grid_size, grid_size), dtype=bool)
        
        self.current_step = 0
        self.total_reward = 0.0
        
        # Pygame
        self.screen = None
        self.clock = None
        self.font = None
        
        # Initialize walls
        self._create_walls()
    
    def _create_walls(self) -> None:
        """Create wall layout based on configuration."""
        self.walls = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        
        # Boundary walls
        self.walls[0, :] = True
        self.walls[-1, :] = True
        self.walls[:, 0] = True
        self.walls[:, -1] = True
        
        if self.wall_layout == "simple":
            # Simple interior walls
            self.walls[30, 20:50] = True
            self.walls[50:80, 70] = True
            self.walls[70:85, 30] = True
            self.walls[70, 15:31] = True
            
        elif self.wall_layout == "complex":
            # Complex room-like layout
            self.walls[25, 10:40] = True
            self.walls[25, 60:90] = True
            self.walls[50, 20:45] = True
            self.walls[50, 55:80] = True
            self.walls[75, 10:35] = True
            self.walls[75, 65:90] = True
            self.walls[10:40, 50] = True
            self.walls[60:90, 50] = True
    
    def _is_valid_position(self, x: float, y: float, radius: float = ROBOT_RADIUS) -> bool:
        """Check if position is valid (not colliding with walls)."""
        # Boundary check
        if x < radius or x > self.grid_size - radius:
            return False
        if y < radius or y > self.grid_size - radius:
            return False
        
        # Wall collision check
        for dx in [-radius, 0, radius]:
            for dy in [-radius, 0, radius]:
                check_x = int(np.clip(x + dx, 0, self.grid_size - 1))
                check_y = int(np.clip(y + dy, 0, self.grid_size - 1))
                if self.walls[check_x, check_y]:
                    return False
        
        return True
    
    def _spawn_entity_random(self, radius: float = ROBOT_RADIUS) -> Tuple[float, float]:
        """Spawn an entity at a random valid position."""
        for _ in range(100):
            x = np.random.uniform(radius + 5, self.grid_size - radius - 5)
            y = np.random.uniform(radius + 5, self.grid_size - radius - 5)
            if self._is_valid_position(x, y, radius):
                return x, y
        return self.grid_size / 2, self.grid_size / 2
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.total_reward = 0.0
        
        # Reset robots
        self.robots = []
        for i in range(self.num_robots):
            x, y = self._spawn_entity_random(ROBOT_RADIUS)
            self.robots.append(RobotState(
                id=i, x=x, y=y,heading=np.random.uniform(0, 2 * np.pi)
            ))
        
        # Reset humans
        self.humans = []
        patterns = list(HumanMovementPattern)
        for i in range(self.num_humans):
            x, y = self._spawn_entity_random(HUMAN_RADIUS)
            pattern = patterns[i % len(patterns)]
            patrol_points = []
            if pattern == HumanMovementPattern.LINEAR_PATROL:
                patrol_points = [
                    (x, y),
                    (x + np.random.uniform(-20, 20), y + np.random.uniform(-20, 20))
                ]
            
            self.humans.append(HumanState(
                id=i, x=x, y=y, pattern=pattern,
                patrol_points=patrol_points, speed=np.random.uniform(0.3, 1.0)
            ))
        
        # Reset targets
        self.targets = []
        for i in range(self.num_targets):
            x, y = self._spawn_entity_random(TARGET_RADIUS)
            self.targets.append(TargetState(id=i, x=x, y=y))
        
        return self.get_observation(0), self._get_info()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment."""
        self.current_step += 1
        
        # Scale actions
        linear_vel = float(action[0]) * MAX_LINEAR_VELOCITY
        angular_vel = float(action[1]) * MAX_ANGULAR_VELOCITY
        
        # Move primary robot
        robot = self.robots[0]
        reward = self._move_robot(robot, linear_vel, angular_vel)
        
        # Move secondary robots
        if not getattr(self, 'disable_secondary_auto', False):
            for i in range(1, len(self.robots)):
                self._move_robot_simple(self.robots[i])
        
        # Update humans
        self._update_humans()
        
        # Check termination
        all_targets_reached = all(t.reached for t in self.targets)
        terminated = all_targets_reached
        truncated = self.current_step >= self.max_steps
        
        # Time penalty
        reward += REWARD_TIME_PENALTY
        self.total_reward += reward
        
        return self.get_observation(0), reward, terminated, truncated, self._get_info()
    
    def _move_robot(self, robot: RobotState, linear_vel: float, angular_vel: float) -> float:
        """Move a robot and calculate reward."""
        reward = 0.0
        
        robot.prev_velocity = robot.velocity
        robot.velocity = linear_vel
        
        # Rewards/Penalties
        if abs(linear_vel) > 0.5: reward += REWARD_FORWARD_MOVEMENT
        if abs(angular_vel) > 0.8 and abs(linear_vel) < 0.3: reward += REWARD_SPINNING_PENALTY
        
        # Update position
        robot.heading = (robot.heading + angular_vel) % (2 * np.pi)
        new_x = robot.x + linear_vel * np.cos(robot.heading)
        new_y = robot.y + linear_vel * np.sin(robot.heading)
        
        # Collision Check
        if not self._is_valid_position(new_x, new_y, ROBOT_RADIUS):
            reward += REWARD_WALL_COLLISION
            robot.collisions += 1
            robot.wall_collisions += 1
            # Nudge
            robot.x -= linear_vel * np.cos(robot.heading) * 0.2
            robot.y -= linear_vel * np.sin(robot.heading) * 0.2
            robot.heading = (robot.heading + np.random.choice([-1.5, 1.5])) % (2 * np.pi)
        else:
            robot.x = new_x
            robot.y = new_y
        
        # Path history
        robot.path_history.append((robot.x, robot.y))
        if len(robot.path_history) > 200: robot.path_history.pop(0)
        
        # Human Collision
        robot_pos = np.array([robot.x, robot.y])
        for human in self.humans:
            if np.linalg.norm(robot_pos - np.array([human.x, human.y])) < (ROBOT_RADIUS + HUMAN_RADIUS):
                reward += REWARD_HUMAN_COLLISION
                robot.collisions += 1
        
        # Targets
        min_target_dist = float('inf')
        for target in self.targets:
            if not target.reached:
                dist = np.linalg.norm(robot_pos - np.array([target.x, target.y]))
                min_target_dist = min(min_target_dist, dist)
                if dist < TARGET_RADIUS:
                    target.reached = True
                    reward += REWARD_TARGET_REACHED
                    robot.targets_reached += 1
        
        if min_target_dist != float('inf'):
            reward += REWARD_DISTANCE_PENALTY * (min_target_dist / self.grid_size)
        
        # Social Cost
        human_positions = np.array([[h.x, h.y] for h in self.humans])
        social_cost = calculate_social_cost(robot_pos, human_positions)
        reward += REWARD_SOCIAL_FRICTION * min(social_cost, 5.0)
        
        return reward
    
    def _move_robot_simple(self, robot: RobotState) -> None:
        """Simple logic for other robots."""
        robot_pos = np.array([robot.x, robot.y])
        targets = [t for t in self.targets if not t.reached]
        if not targets: return
        
        closest = min(targets, key=lambda t: np.linalg.norm(robot_pos - np.array([t.x, t.y])))
        direction = np.array([closest.x - robot.x, closest.y - robot.y])
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction /= norm
            new_x, new_y = robot.x + direction[0] * 0.5, robot.y + direction[1] * 0.5
            if self._is_valid_position(new_x, new_y, ROBOT_RADIUS):
                robot.x, robot.y = new_x, new_y

    def _update_humans(self) -> None:
        """Update human positions."""
        for human in self.humans:
            # Simplified movement logic for brevity - keeps them moving
            if human.pattern == HumanMovementPattern.RANDOM_WALK:
                human.vx += np.random.normal(0, 0.2)
                human.vy += np.random.normal(0, 0.2)
                speed = np.sqrt(human.vx**2 + human.vy**2)
                if speed > human.speed:
                    human.vx = (human.vx / speed) * human.speed
                    human.vy = (human.vy / speed) * human.speed
            
            new_x, new_y = human.x + human.vx, human.y + human.vy
            
            # Bounce
            if not self._is_valid_position(new_x, new_y, HUMAN_RADIUS):
                human.vx *= -1; human.vy *= -1
                new_x, new_y = human.x + human.vx, human.y + human.vy
            
            if self._is_valid_position(new_x, new_y, HUMAN_RADIUS):
                human.x, human.y = new_x, new_y

    def _get_lidar_scan(self, robot: RobotState) -> np.ndarray:
        """Simulate lidar."""
        scan = np.full(LIDAR_RAYS, LIDAR_MAX_RANGE)
        for i in range(LIDAR_RAYS):
            angle = robot.heading + (2 * np.pi * i / LIDAR_RAYS)
            for dist in np.linspace(1, LIDAR_MAX_RANGE, 10): # Lower res for speed
                cx = robot.x + dist * np.cos(angle)
                cy = robot.y + dist * np.sin(angle)
                if not self._is_valid_position(cx, cy, 0.5):
                    scan[i] = dist
                    break
        return scan / LIDAR_MAX_RANGE

    def get_observation(self, robot_idx: int = 0) -> np.ndarray:
        """Get observation vector."""
        robot = self.robots[robot_idx]
        lidar = self._get_lidar_scan(robot)
        
        # Human Rel
        human_rel = []
        rpos = np.array([robot.x, robot.y])
        for i in range(8):
            if i < len(self.humans):
                hpos = np.array([self.humans[i].x, self.humans[i].y])
                human_rel.extend((hpos - rpos) / self.grid_size)
            else:
                human_rel.extend([0.0, 0.0])
        
        # Target Rel
        target_rel = np.array([0.0, 0.0])
        targets = [t for t in self.targets if not t.reached]
        if targets:
            closest = min(targets, key=lambda t: np.linalg.norm(rpos - np.array([t.x, t.y])))
            target_rel = (np.array([closest.x, closest.y]) - rpos) / self.grid_size
            
        crowd = min(calculate_social_cost(rpos, np.array([[h.x, h.y] for h in self.humans])) / 10.0, 1.0)
        
        return np.concatenate([
            [robot.x/self.grid_size, robot.y/self.grid_size, robot.heading/(2*np.pi)],
            lidar,
            human_rel,
            target_rel,
            [crowd],
            [robot.battery/100.0]
        ]).astype(np.float32)

    def _get_info(self) -> Dict[str, Any]:
        robot = self.robots[0]
        return {
            'step': self.current_step,
            'total_reward': self.total_reward,
            'robot_position': (robot.x, robot.y),
            'targets_reached': self.robots[0].targets_reached,
            'total_targets': self.num_targets,
            'collisions': robot.collisions
        }

    def render(self) -> Optional[np.ndarray]:
        """Render using Pygame with Overlay Legend."""
        if self.screen is None and self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((self.grid_size * CELL_SIZE, self.grid_size * CELL_SIZE))
            pygame.display.set_caption("Multi-Robot DDPG Environment")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 24)
            
        if self.screen is None: return None
        
        self.screen.fill(COLORS['background'])
        
        # Draw Walls
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.walls[i, j]:
                    pygame.draw.rect(self.screen, COLORS['wall'], (i*CELL_SIZE, j*CELL_SIZE, CELL_SIZE, CELL_SIZE))
        
        # Draw Targets
        for t in self.targets:
            color = COLORS['target_reached'] if t.reached else COLORS['target']
            pygame.draw.circle(self.screen, color, (int(t.x*CELL_SIZE), int(t.y*CELL_SIZE)), int(TARGET_RADIUS*CELL_SIZE))
            
        # Draw Humans
        for h in self.humans:
            pygame.draw.circle(self.screen, COLORS['human'], (int(h.x*CELL_SIZE), int(h.y*CELL_SIZE)), int(HUMAN_RADIUS*CELL_SIZE))
            
        # Draw Robots
        for r in self.robots:
            if len(r.path_history) > 1:
                points = [(int(p[0]*CELL_SIZE), int(p[1]*CELL_SIZE)) for p in r.path_history[-50:]]
                pygame.draw.lines(self.screen, COLORS['path'], False, points, 2)
            pygame.draw.circle(self.screen, COLORS['robot'], (int(r.x*CELL_SIZE), int(r.y*CELL_SIZE)), int(ROBOT_RADIUS*CELL_SIZE))
            
        self._draw_overlay_legend()
        
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])
        return None

    def _draw_overlay_legend(self):
        """Draw top-left overlay legend."""
        padding = 10
        item_height = 25
        width = 200
        height = 5 * item_height + 2 * padding
        x, y = 10, 10
        
        s = pygame.Surface((width, height))
        s.set_alpha(200); s.fill((30, 30, 40))
        self.screen.blit(s, (x, y))
        pygame.draw.rect(self.screen, (100, 100, 100), (x, y, width, height), 2)
        
        font = pygame.font.SysFont("Arial", 16, bold=True)
        items = [
            (COLORS['robot'], "Robot (Agent)"),
            (COLORS['target'], "Target (Goal)"),
            (COLORS['human'], "Human (Avoid)"),
            (COLORS['wall'], "Wall (Block)"),
            (COLORS['path'], "Path Trail"),
        ]
        
        for i, (col, lbl) in enumerate(items):
            iy = y + padding + i*item_height
            pygame.draw.circle(self.screen, col, (x+20, iy+8), 8)
            pygame.draw.circle(self.screen, (255,255,255), (x+20, iy+8), 8, 1)
            self.screen.blit(font.render(lbl, True, (255,255,255)), (x+40, iy))

    def close(self):
        if self.screen: pygame.quit(); self.screen = None
