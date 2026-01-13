"""
Social Cost Module
==================
Calculates social friction costs for human-aware robot navigation.
Uses inverse-square law for proximity penalties to encourage robots
to maintain safe distances from humans.
"""

import numpy as np
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================================
# CONSTANTS
# ============================================================================

INFLUENCE_RADIUS = 5.0      # Meters - radius of human influence
MIN_DISTANCE = 0.1          # Minimum distance to prevent division by zero
SOCIAL_WEIGHT = 2.0         # Weight for social cost in task allocation


# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def calculate_social_cost(
    robot_pos: np.ndarray,
    human_positions: np.ndarray,
    influence_radius: float = INFLUENCE_RADIUS
) -> float:
    """
    Calculate crowd density around robot position using inverse-square law.
    
    Higher values indicate more crowded areas that should be avoided.
    
    Args:
        robot_pos: Robot position [x, y]
        human_positions: Array of human positions [[x1, y1], [x2, y2], ...]
        influence_radius: Maximum distance for human influence
        
    Returns:
        Social cost value (higher = more crowded)
        
    Example:
        >>> robot = np.array([50, 50])
        >>> humans = np.array([[52, 50], [48, 51]])
        >>> cost = calculate_social_cost(robot, humans)
        >>> print(f"Social cost: {cost:.2f}")
    """
    if len(human_positions) == 0:
        return 0.0
    
    density = 0.0
    for human_pos in human_positions:
        distance = np.linalg.norm(robot_pos - human_pos)
        
        if distance < influence_radius:
            # Inverse square law - closer humans have higher influence
            density += 1.0 / (distance + MIN_DISTANCE) ** 2
    
    return density


def calculate_path_social_cost(
    path: np.ndarray,
    human_positions: np.ndarray,
    influence_radius: float = INFLUENCE_RADIUS
) -> float:
    """
    Calculate total social cost along a path.
    
    Args:
        path: Array of waypoints [[x1, y1], [x2, y2], ...]
        human_positions: Array of human positions
        influence_radius: Maximum distance for human influence
        
    Returns:
        Total social cost along the path
    """
    total_cost = 0.0
    for waypoint in path:
        total_cost += calculate_social_cost(waypoint, human_positions, influence_radius)
    return total_cost / len(path) if len(path) > 0 else 0.0


def get_social_friction_map(
    human_positions: np.ndarray,
    grid_size: int = 100,
    influence_radius: float = INFLUENCE_RADIUS
) -> np.ndarray:
    """
    Create 2D heatmap of crowd density across the entire grid.
    
    Useful for visualization and path planning.
    
    Args:
        human_positions: Array of human positions [[x1, y1], ...]
        grid_size: Size of the grid (grid_size x grid_size)
        influence_radius: Maximum distance for human influence
        
    Returns:
        2D numpy array of social friction values
        
    Example:
        >>> humans = np.array([[25, 25], [75, 75], [50, 30]])
        >>> friction_map = get_social_friction_map(humans, grid_size=100)
        >>> print(f"Map shape: {friction_map.shape}")
    """
    friction_map = np.zeros((grid_size, grid_size))
    
    for i in range(grid_size):
        for j in range(grid_size):
            pos = np.array([i, j], dtype=np.float32)
            friction_map[i, j] = calculate_social_cost(
                pos, human_positions, influence_radius
            )
    
    return friction_map


def get_local_density(
    position: np.ndarray,
    human_positions: np.ndarray,
    radius: float = 10.0
) -> int:
    """
    Count number of humans within a given radius.
    
    Args:
        position: Query position [x, y]
        human_positions: Array of human positions
        radius: Search radius
        
    Returns:
        Number of humans within radius
    """
    if len(human_positions) == 0:
        return 0
    
    distances = np.linalg.norm(human_positions - position, axis=1)
    return int(np.sum(distances < radius))


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_social_friction_map(
    friction_map: np.ndarray,
    human_positions: Optional[np.ndarray] = None,
    robot_positions: Optional[np.ndarray] = None,
    target_positions: Optional[np.ndarray] = None,
    title: str = "Social Friction Heatmap",
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Visualize the social friction map with optional overlays.
    
    Creates a beautiful heatmap showing crowd density with markers
    for humans, robots, and targets.
    
    Args:
        friction_map: 2D array of friction values
        human_positions: Optional human positions to overlay
        robot_positions: Optional robot positions to overlay
        target_positions: Optional target positions to overlay
        title: Plot title
        save_path: Path to save the figure (optional)
        show: Whether to display the plot
        
    Returns:
        matplotlib Figure object
    """
    # Set up the figure with a dark theme
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Create heatmap with custom colormap
    cmap = sns.color_palette("YlOrRd", as_cmap=True)
    im = ax.imshow(
        friction_map.T, 
        origin='lower', 
        cmap=cmap, 
        interpolation='bilinear',
        alpha=0.8
    )
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, label='Social Friction')
    cbar.ax.tick_params(labelsize=10)
    
    # Plot humans as red circles
    if human_positions is not None and len(human_positions) > 0:
        ax.scatter(
            human_positions[:, 0], 
            human_positions[:, 1],
            c='#FF6B6B', 
            s=200, 
            marker='o',
            edgecolors='white',
            linewidths=2,
            label='Humans',
            zorder=5
        )
    
    # Plot robots as cyan squares
    if robot_positions is not None and len(robot_positions) > 0:
        ax.scatter(
            robot_positions[:, 0], 
            robot_positions[:, 1],
            c='#4ECDC4', 
            s=250, 
            marker='s',
            edgecolors='white',
            linewidths=2,
            label='Robots',
            zorder=6
        )
    
    # Plot targets as green stars
    if target_positions is not None and len(target_positions) > 0:
        ax.scatter(
            target_positions[:, 0], 
            target_positions[:, 1],
            c='#95E77A', 
            s=300, 
            marker='*',
            edgecolors='white',
            linewidths=2,
            label='Targets',
            zorder=7
        )
    
    # Styling
    ax.set_xlabel('X Position', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y Position', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_aspect('equal')
    
    # Grid
    ax.grid(True, alpha=0.2, linestyle='--')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    
    if show:
        plt.show()
    
    return fig


# ============================================================================
# MAIN (Testing)
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Social Cost Module - Test")
    print("=" * 60)
    
    # Create test scenario
    np.random.seed(42)
    humans = np.random.uniform(20, 80, size=(6, 2))
    robots = np.array([[30, 30], [70, 70]])
    targets = np.array([[50, 20], [20, 80], [80, 50]])
    
    # Calculate friction map
    print("\n📊 Generating social friction map...")
    friction_map = get_social_friction_map(humans, grid_size=100)
    print(f"   Map shape: {friction_map.shape}")
    print(f"   Max friction: {friction_map.max():.2f}")
    print(f"   Mean friction: {friction_map.mean():.4f}")
    
    # Test individual cost
    robot_pos = np.array([50, 50])
    cost = calculate_social_cost(robot_pos, humans)
    print(f"\n🤖 Social cost at center: {cost:.2f}")
    
    # Visualize
    print("\n🎨 Creating visualization...")
    plot_social_friction_map(
        friction_map,
        human_positions=humans,
        robot_positions=robots,
        target_positions=targets,
        title="Social Friction Map - Test Scenario"
    )
    
    print("\n✅ Test complete!")
