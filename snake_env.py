import pygame
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque
import random
from enum import Enum
import time # For human play speed control

# --- Game Configuration ---
GRID_SIZE = 15
CELL_SIZE = 30 # For rendering
WINDOW_WIDTH = GRID_SIZE * CELL_SIZE
WINDOW_HEIGHT = GRID_SIZE * CELL_SIZE + 40 # Extra space for score

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (213, 50, 80)      # Food
GREEN_HEAD = (0, 150, 0) # Snake Head
GREEN_BODY = (0, 255, 0) # Snake Body
BLUE_SCORE = (50, 50, 213) # Score Text

# Game Speed (for human/agent-play modes)
INITIAL_SPEED = 10

# --- Game Objects ---
class Direction(Enum):
    RIGHT = 0
    LEFT = 1
    UP = 2
    DOWN = 3

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash((self.x, self.y))

    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y)

# --- Gymnasium Environment ---
class SnakeEnv(gym.Env):
    """
    Custom Environment for Snake game that follows gym interface.

    Observation Space:
    A vector containing information about the snake's surroundings,
    its direction, and the relative position of the food.
    Features:
    1. Danger Straight (1.0 if wall/body 1 step ahead, 0.0 otherwise)
    2. Danger Left     (1.0 if wall/body 1 step left relative to direction)
    3. Danger Right    (1.0 if wall/body 1 step right relative to direction)
    4. Direction Left  (1.0 if current direction is Left)
    5. Direction Right (1.0 if current direction is Right)
    6. Direction Up    (1.0 if current direction is Up)
    7. Direction Down  (1.0 if current direction is Down)
    8. Food Delta X    (Normalized difference in x: (food.x - head.x) / GRID_SIZE)
    9. Food Delta Y    (Normalized difference in y: (food.y - head.y) / GRID_SIZE)
    10. Tail Delta X   (Normalized difference in x: (tail.x - head.x) / GRID_SIZE) - Helps avoid cycles
    11. Tail Delta Y   (Normalized difference in y: (tail.y - head.y) / GRID_SIZE) - Helps avoid cycles

    Action Space:
    Discrete(4) representing UP, DOWN, LEFT, RIGHT.
    Note: The environment internally prevents the snake from reversing direction.

    Reward Structure:
    - +10.0 for eating food.
    - -10.0 for dying (collision with wall or self).
    - +0.1 for moving closer to the food.
    - -0.15 for moving further away from the food.
    - -0.01 penalty per step to encourage efficiency.

    Termination:
    The episode ends if the snake collides with the wall or itself.

    Truncation:
    The episode can be truncated if it exceeds a maximum number of steps
    without eating food, preventing infinite loops in sparse reward scenarios.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": INITIAL_SPEED}

    def __init__(self, render_mode=None, grid_size=GRID_SIZE, max_steps_no_food=GRID_SIZE*GRID_SIZE*2):
        super().__init__()

        self.grid_size = grid_size
        self.window_width = self.grid_size * CELL_SIZE
        self.window_height = self.grid_size * CELL_SIZE + 40
        self.max_steps_no_food = max_steps_no_food # Truncation limit

        # Define action and observation space
        # Actions: 0: Right, 1: Left, 2: Up, 3: Down (matches Direction Enum)
        self.action_space = spaces.Discrete(4)

        # Observation space: See docstring for details
        # Using Box for continuous/binary features. Normalized values where applicable.
        # Danger flags (3), Direction flags (4), Food delta (2), Tail delta (2)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(11,), dtype=np.float32)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # Pygame setup (only if rendering)
        self.window = None
        self.clock = None
        self.font = None
        if self.render_mode == "human":
            pygame.init()
            self.font = pygame.font.SysFont('arial', 25)
            self.window = pygame.display.set_mode((self.window_width, self.window_height))
            pygame.display.set_caption('RL Snake')
            self.clock = pygame.time.Clock()

        # Internal game state is reset in reset()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed) # Important for reproducibility

        # Initialize game state
        self.direction = random.choice(list(Direction)) # Start random direction

        # Start snake in the center
        start_x = self.grid_size // 2
        start_y = self.grid_size // 2
        self.head = Point(start_x, start_y)
        self.snake = deque([
            self.head,
            # Add initial body segments based on starting direction
            Point(self.head.x - (1 if self.direction == Direction.RIGHT else -1 if self.direction == Direction.LEFT else 0),
                  self.head.y - (1 if self.direction == Direction.DOWN else -1 if self.direction == Direction.UP else 0))
        ])
        self._snake_set = set(self.snake) # For faster collision checks

        self.score = 0
        self.steps_since_last_food = 0
        self.total_steps = 0

        self._place_food()

        observation = self._get_observation()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def _place_food(self):
        while True:
            x = self.np_random.integers(0, self.grid_size)
            y = self.np_random.integers(0, self.grid_size)
            self.food = Point(x, y)
            if self.food not in self._snake_set:
                break

    def _get_info(self):
        # Provide additional info, useful for debugging or analysis
        return {"score": self.score, "length": len(self.snake)}

    def _get_observation(self):
        # --- Calculate Danger ---
        # Points relative to the current direction
        point_ahead = self._get_next_head_position(self.direction)
        point_left_rel, point_right_rel = self._get_relative_points(self.direction)

        danger_straight = self._is_collision(point_ahead)
        danger_left = self._is_collision(self.head + point_left_rel)
        danger_right = self._is_collision(self.head + point_right_rel)

        # --- Direction Flags (One-Hot Encoding) ---
        dir_l = self.direction == Direction.LEFT
        dir_r = self.direction == Direction.RIGHT
        dir_u = self.direction == Direction.UP
        dir_d = self.direction == Direction.DOWN

        # --- Food Position ---
        # Normalized relative coordinates
        food_delta_x = (self.food.x - self.head.x) / self.grid_size
        food_delta_y = (self.food.y - self.head.y) / self.grid_size

        # --- Tail Position ---
        # Normalized relative coordinates of the tail
        tail = self.snake[-1]
        tail_delta_x = (tail.x - self.head.x) / self.grid_size
        tail_delta_y = (tail.y - self.head.y) / self.grid_size


        observation = np.array([
            danger_straight,
            danger_left,
            danger_right,
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            food_delta_x,
            food_delta_y,
            tail_delta_x,
            tail_delta_y
        ], dtype=np.float32)

        return observation

    def _get_relative_points(self, current_direction):
        # Returns points representing 'left' and 'right' relative to the current direction
        # Clockwise turning logic
        if current_direction == Direction.UP:
            return Point(-1, 0), Point(1, 0) # Left, Right
        elif current_direction == Direction.DOWN:
            return Point(1, 0), Point(-1, 0) # Left, Right
        elif current_direction == Direction.LEFT:
            return Point(0, 1), Point(0, -1) # Left (Down), Right (Up)
        elif current_direction == Direction.RIGHT:
            return Point(0, -1), Point(0, 1) # Left (Up), Right (Down)

    def _get_next_head_position(self, direction):
        # Calculates the potential next position of the head based on a direction
        if direction == Direction.RIGHT:
            return self.head + Point(1, 0)
        elif direction == Direction.LEFT:
            return self.head + Point(-1, 0)
        elif direction == Direction.DOWN:
            return self.head + Point(0, 1)
        elif direction == Direction.UP:
            return self.head + Point(0, -1)

    def step(self, action):
        # --- Determine new direction based on action ---
        new_direction = Direction(action)

        # Prevent snake from reversing direction
        opposite_direction = {
            Direction.UP: Direction.DOWN,
            Direction.DOWN: Direction.UP,
            Direction.LEFT: Direction.RIGHT,
            Direction.RIGHT: Direction.LEFT
        }
        if len(self.snake) > 1 and new_direction == opposite_direction[self.direction]:
             # If trying to reverse, continue in the current direction
             new_direction = self.direction
        self.direction = new_direction

        # --- Move snake ---
        next_head = self._get_next_head_position(self.direction)
        prev_head = self.head # Store old head for distance calculation

        # --- Calculate Reward Components ---
        reward = -0.01 # Small penalty for existing per step

        # Distance-based reward (before moving head)
        dist_before = abs(self.head.x - self.food.x) + abs(self.head.y - self.food.y)
        dist_after = abs(next_head.x - self.food.x) + abs(next_head.y - self.food.y)

        if dist_after < dist_before:
            reward += 0.1 # Reward for getting closer
        else:
            # Penalize slightly more for moving away than just existing
            reward -= 0.15 # Penalty for moving further away


        # --- Check for collisions ---
        terminated = False
        if self._is_collision(next_head):
            terminated = True
            reward = -10.0 # Large penalty for dying
            # print(f"Collision! Final Score: {self.score}") # Debug
        else:
            # Update head position
            self.head = next_head
            self.snake.appendleft(self.head)
            self._snake_set.add(self.head) # Add new head to set

            # --- Check for food consumption ---
            if self.head == self.food:
                self.score += 1
                reward = 10.0 # Large reward for eating food
                self._place_food()
                self.steps_since_last_food = 0
                # Don't remove tail segment when food is eaten (snake grows)
            else:
                # Remove tail segment if no food was eaten
                tail = self.snake.pop()
                self._snake_set.remove(tail) # Remove tail from set
                self.steps_since_last_food += 1

        # --- Check for truncation ---
        truncated = False
        if self.steps_since_last_food > self.max_steps_no_food:
            truncated = True
            # Optional: Add a small penalty for truncation if desired
            # reward -= 1.0
            # print(f"Truncated! Steps without food: {self.steps_since_last_food}") # Debug

        self.total_steps += 1

        # --- Get new state ---
        observation = self._get_observation()
        info = self._get_info()

        # --- Render ---
        if self.render_mode == "human":
            self._render_frame()
            # Control speed for human play/viewing
            self.clock.tick(self.metadata["render_fps"])

        return observation, reward, terminated, truncated, info

    def _is_collision(self, point):
        # Check wall collision
        if point.x >= self.grid_size or point.x < 0 or point.y >= self.grid_size or point.y < 0:
            return True
        # Check self collision (excluding the very tail in the next step if not growing)
        if point in self._snake_set:
             # Allow moving into the tail's position *only if* the snake is about to move
             # and therefore vacate that position. This check is implicitly handled by
             # adding the new head *before* removing the tail in the step function.
             # So, if point is in _snake_set, it's a collision.
            return True

        return False

    def render(self):
        # The main rendering logic is called within step() for "human" mode
        # For "rgb_array" mode, we need to generate the image data
        if self.render_mode == "rgb_array":
            return self._render_frame(to_rgb_array=True)
        # If render_mode is None, do nothing extra here.

    def _render_frame(self, to_rgb_array=False):
        if self.window is None and self.render_mode == "human":
            # Initialize pygame if not already done (e.g., if reset wasn't called with human mode)
            pygame.init()
            self.font = pygame.font.SysFont('arial', 25)
            self.window = pygame.display.set_mode((self.window_width, self.window_height))
            pygame.display.set_caption('RL Snake')
            self.clock = pygame.time.Clock()

        if self.clock is None and self.render_mode == "human":
             self.clock = pygame.time.Clock()

        # Create a surface to draw on
        canvas = pygame.Surface((self.window_width, self.window_height))
        canvas.fill(BLACK)

        # Draw Grid lines (optional, can make it visually clearer)
        # for x in range(0, self.window_width, CELL_SIZE):
        #     pygame.draw.line(canvas, WHITE, (x, 0), (x, self.window_height - 40), 1)
        # for y in range(0, self.window_height - 40, CELL_SIZE):
        #     pygame.draw.line(canvas, WHITE, (0, y), (self.window_width, y), 1)


        # Draw snake
        for i, pt in enumerate(self.snake):
            color = GREEN_HEAD if i == 0 else GREEN_BODY
            pygame.draw.rect(canvas, color, pygame.Rect(pt.x * CELL_SIZE, pt.y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
            # Optional: Add inner rect for visual style
            pygame.draw.rect(canvas, BLACK, pygame.Rect(pt.x * CELL_SIZE + 4, pt.y * CELL_SIZE + 4, CELL_SIZE - 8, CELL_SIZE - 8))


        # Draw food
        pygame.draw.rect(canvas, RED, pygame.Rect(self.food.x * CELL_SIZE, self.food.y * CELL_SIZE, CELL_SIZE, CELL_SIZE))

        # Draw score
        score_text = self.font.render(f"Score: {self.score}", True, BLUE_SCORE)
        canvas.blit(score_text, [10, self.window_height - 35])

        if self.render_mode == "human":
            # Update the display
            self.window.blit(canvas, (0, 0))
            pygame.event.pump() # Process event queue
            pygame.display.flip()

        elif to_rgb_array:
             # Convert Pygame surface to numpy array for rgb_array mode
             return np.transpose(
                 np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
             )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None # Ensure it's reset for potential re-init

# --- Helper function for human play ---
def get_human_action(current_direction):
    action = None
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return "QUIT"
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                action = Direction.LEFT
            elif event.key == pygame.K_RIGHT:
                action = Direction.RIGHT
            elif event.key == pygame.K_UP:
                action = Direction.UP
            elif event.key == pygame.K_DOWN:
                action = Direction.DOWN
            elif event.key == pygame.K_q: # Quit key
                 return "QUIT"

    # Prevent reversing if an action was chosen
    if action is not None:
        opposite_direction = {
            Direction.UP: Direction.DOWN, Direction.DOWN: Direction.UP,
            Direction.LEFT: Direction.RIGHT, Direction.RIGHT: Direction.LEFT
        }
        # Check if snake length > 1 to allow reversal at start
        # This logic is handled better inside env.step, but good for direct human control feel
        # if len(env.snake) > 1 and action == opposite_direction[current_direction]:
        #     return None # Ignore reverse input, continue current direction implicitly

        return action.value # Return the integer value for the environment step

    return None # No valid key pressed or trying to reverse