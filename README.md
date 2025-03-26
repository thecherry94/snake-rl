# RL Snake Game Tutorial

This project demonstrates Reinforcement Learning (RL) concepts using the classic Snake game. It utilizes Python with `pygame` for visualization, `gymnasium` for the environment interface, and `stable-baselines3` for implementing RL algorithms.

**Target Audience:** Individuals with some Machine Learning background who want a practical refresher on Reinforcement Learning principles.

## Features

*   **Classic Snake Game:** Implemented using `pygame`.
*   **Gymnasium Environment:** A custom `SnakeEnv` class compatible with the `gymnasium` API.
*   **Three Modes:**
    1.  `human`: Play the game yourself using arrow keys.
    2.  `train`: Train an RL agent (PPO, DQN, or A2C) to play Snake. Training is optimized (no rendering by default) and uses vectorized environments for speed.
    3.  `agent-play`: Watch a pre-trained agent play the game at a configurable speed.
*   **Clear Visuals:** Distinct colors and score display for easy game state understanding.
*   **RL Concept Explanations:** Code comments and this README explain key RL terms like Environment, Agent, Policy, and Reward Shaping in the context of the Snake game.
*   **Sophisticated Agent Goal:** The observation space and reward structure are designed to encourage the agent to learn intelligent behaviors like avoiding self-collision and navigating efficiently, even as the snake grows longer.

## Project Structure

```
rl_snake/
├── snake_env.py       # Defines the Gymnasium environment for the Snake game
├── main.py            # Main script to run different modes
├── requirements.txt   # Lists necessary libraries
├── models/            # Directory to save trained models
├── logs/              # Directory for TensorBoard logs
└── README.md          # This file
```

## Setup

1.  **Clone the repository (or create the files):**
    ```bash
    git clone <your-repo-url> # Or copy the code into the structure above
    cd rl_snake
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: This installs `stable-baselines3[extra]` which includes `tensorboard`. It also installs either `torch` or `tensorflow` depending on your SB3 backend choice during installation or if already present.)*

## Usage

The main script `main.py` uses command-line arguments to select the mode and configure options.

```bash
python main.py --mode <MODE> [OPTIONS]
```

**Modes:**

*   `--mode human`: Play the game yourself.
    *   `--speed <FPS>`: Set the game speed (frames per second, default: 10).
*   `--mode train`: Train an RL agent.
    *   `--algo <ppo|dqn|a2c>`: Choose the RL algorithm (default: `ppo`).
    *   `--timesteps <N>`: Set the total number of training steps (default: 100,000). More steps generally lead to better agents.
    *   `--envs <N>`: Number of parallel environments to use for faster training (default: 4).
    *   `--load <path/to/model.zip>`: (Optional) Continue training from a previously saved model.
    *   `--save-name <name>`: (Optional) Custom base name for saved models and logs (defaults to `<algo>_snake`).
*   `--mode agent-play`: Watch a trained agent play.
    *   `--load <path/to/model.zip>`: **Required** (or defaults to `models/<algo>_snake_best/best_model.zip` or `models/<algo>_snake_final.zip`). Path to the saved agent model.
    *   `--algo <ppo|dqn|a2c>`: Specify the algorithm the model was trained with (helps load correctly, default: `ppo`).
    *   `--speed <FPS>`: Set the playback speed (default: 10).

**Examples:**

```bash
# Play the game yourself
python main.py --mode human --speed 15

# Train a PPO agent for 500,000 steps using 8 environments
python main.py --mode train --algo ppo --timesteps 500000 --envs 8 --save-name ppo_snake_long_run

# Train a DQN agent, continuing from a previous checkpoint
python main.py --mode train --algo dqn --timesteps 200000 --load models/dqn_snake_checkpoints/rl_model_100000_steps.zip

# Watch the best trained PPO agent play (automatically finds best/final model)
python main.py --mode agent-play --algo ppo --speed 12

# Watch a specific agent model play
python main.py --mode agent-play --load models/ppo_snake_final.zip --speed 8
```

## Reinforcement Learning Concepts Explained

*(See the `RL_EXPLANATION` string within `main.py` for a detailed breakdown integrated into the training output, covering Environment, Agent, Policy, Reward, Learning, and Credit Assignment.)*

**Key Components in this Project:**

1.  **Environment (`SnakeEnv` in `snake_env.py`):**
    *   **Observation Space:** A vector describing the snake's immediate surroundings (danger left/straight/right), its current direction, and the relative direction to the food and its own tail. This provides the agent with the necessary context to make informed decisions.
    *   **Action Space:** Discrete set of actions: Up, Down, Left, Right.
    *   **Reward Function:** This is critical for learning. The agent receives:
        *   Large positive reward (+10) for eating food.
        *   Large negative reward (-10) for dying.
        *   Small positive/negative rewards (+0.1 / -0.15) for moving towards/away from the food (Reward Shaping).
        *   Small negative penalty (-0.01) per step to encourage efficiency.
    *   **Termination/Truncation:** An episode ends if the snake dies (`terminated=True`) or if it takes too many steps without eating (`truncated=True`).

2.  **Agent/Policy (Managed by Stable Baselines3):**
    *   We use pre-built algorithms like PPO, DQN, or A2C from `stable-baselines3`.
    *   These algorithms use Neural Networks (`MlpPolicy` - Multi-Layer Perceptron) to learn the mapping from Observations to Actions.
    *   During training (`model.learn(...)`), the agent interacts with many copies of the `SnakeEnv`, collects experiences (state, action, reward, next state), and updates its policy network to maximize the expected cumulative future reward, guided by the reward signals from the environment.

3.  **Training Intelligence:**
    *   The observation space includes danger detection and relative food/tail positions to help the agent learn complex behaviors.
    *   The reward shaping encourages efficient paths towards food.
    *   Training for sufficient `timesteps` allows the agent to explore the state space and refine its policy to handle longer snake lengths and avoid common pitfalls like circling or running into itself.

## Monitoring Training (TensorBoard)

Training logs are saved in the `logs/` directory. You can visualize the learning progress (e.g., episode rewards, episode lengths) using TensorBoard:

```bash
tensorboard --logdir logs/
```

Then open your web browser to the URL provided (usually `http://localhost:6006`).

## Further Exploration

*   **Hyperparameter Tuning:** Experiment with different learning rates, network architectures, `gamma` (discount factor), etc., within the SB3 algorithm constructors (`PPO`, `DQN`, `A2C`).
*   **Observation Space:** Try different features. Could adding the full grid (or a portion around the head) as input (using `CnnPolicy`) improve performance? (This would require changing the `observation_space` and likely the policy type in `main.py`).
*   **Reward Shaping:** Modify the reward values. What happens if the penalty for dying is smaller? Or if there's no penalty for moving away from food?
*   **Different Algorithms:** Compare the performance of PPO, DQN, and A2C on this task.
*   **Curriculum Learning:** Start training on a smaller grid and gradually increase the size.