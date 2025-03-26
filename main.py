import gymnasium as gym
import pygame
import argparse
import os
import time

# Import necessary components from stable-baselines3
from stable_baselines3 import PPO, DQN, A2C # Choose your preferred algorithm
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

# Import the custom Snake environment
from snake_env import SnakeEnv, Direction, get_human_action, GRID_SIZE, INITIAL_SPEED

# --- Configuration ---
MODEL_DIR = "models"
LOG_DIR = "logs"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Default model filename
DEFAULT_MODEL_NAME = "ppo_snake" # Change if using DQN or A2C

# --- RL Explanation ---
RL_EXPLANATION = """
=================================
Reinforcement Learning Concepts
=================================

1. Environment (SnakeEnv):
   - Represents the game world (the Snake grid).
   - Defines:
     - State/Observation Space: What the agent 'sees'. In our case, it's a vector
       containing info about danger (walls/body nearby), current direction,
       and relative food/tail location. This needs to capture enough info for
       the agent to make good decisions.
     - Action Space: What the agent can 'do' (Up, Down, Left, Right).
     - Reward Signal: Feedback given to the agent after each action. This guides
       the learning process.
     - Dynamics: How the state changes based on an action (snake moves, eats, dies).

2. Agent (The Trained Model):
   - The learner and decision-maker.
   - Its goal is to maximize the cumulative reward over an episode (a single game).
   - Contains a Policy and optionally a Value Function.

3. Policy (e.g., PPO's 'MlpPolicy'):
   - The agent's strategy or brain. It maps an observation (state) to an action.
   - In Deep RL (like Stable Baselines3 uses), the policy is typically a Neural Network.
   - Input: Observation vector from the environment.
   - Output: Probabilities for each possible action (for PPO/A2C) or Q-values (for DQN).
   - Training adjusts the network's weights to favor actions that lead to higher rewards.

4. Reward Signal (Defined in SnakeEnv.step):
   - Crucial for learning! The agent learns *based on these rewards*.
   - Our Rewards:
     - +10.0: Eating food (Strong positive signal - desired behavior).
     - -10.0: Dying (Strong negative signal - avoid this!).
     - +0.1 / -0.15: Moving towards/away from food (Guides the agent efficiently).
     - -0.01: Per step penalty (Encourages shorter paths, discourages passivity).
   - Reward Shaping: Designing intermediate rewards (like moving towards food) can
     speed up learning, especially when the main reward (eating food) is sparse.

5. Learning Process (e.g., PPO Algorithm):
   - The agent interacts with the environment (plays the game) over many episodes.
   - It collects experiences: (state, action, reward, next_state).
   - The algorithm uses these experiences to update the Policy network.
   - PPO (Proximal Policy Optimization) is an 'on-policy' algorithm. It tries to improve
     the current policy while ensuring the updates don't change the policy too drastically
     (which can destabilize learning). It balances:
     - Exploration: Trying out different actions to discover potentially better strategies.
     - Exploitation: Using the current best-known strategy to maximize rewards.
   - The training process aims to find a policy that consistently achieves high scores.

6. Backtracking / Credit Assignment:
   - A key challenge is figuring out which past actions led to a reward received later.
   - Algorithms use techniques like Value Functions (estimating future rewards from a state)
     and Advantage Estimation (how much better an action was than expected) to assign
     'credit' or 'blame' to actions taken earlier in an episode. This allows the agent
     to learn long-term consequences, not just immediate rewards.
=================================
"""

# --- Mode Functions ---

def play_human():
    """Runs the Snake game in human-playable mode."""
    print("\n--- Human Play Mode ---")
    print("Controls: Arrow Keys (Up, Down, Left, Right)")
    print("Press 'Q' to Quit.")

    # Use render_mode="human" for direct display
    env = SnakeEnv(render_mode="human")
    env.metadata['render_fps'] = INITIAL_SPEED # Control human play speed

    obs, info = env.reset()
    terminated = False
    truncated = False
    total_reward = 0
    current_direction = env.direction # Keep track for input handling

    while True:
        action_val = get_human_action(current_direction) # Get input via Pygame events

        if action_val == "QUIT":
            print("Quitting game.")
            break

        if action_val is None:
            # No valid input, continue in the current direction
            # The environment step function needs the *intended* action based on direction
            action_val = env.direction.value
        else:
            # Update current_direction if a valid move key was pressed
            # Note: env.step internally handles invalid reversals
            current_direction = Direction(action_val)


        obs, reward, terminated, truncated, info = env.step(action_val)
        total_reward += reward

        # print(f"Step: Action={Direction(action_val).name}, Reward={reward:.2f}, Terminated={terminated}, Truncated={truncated}, Score={info['score']}") # Debug Step

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            print("Resetting game...")
            time.sleep(2) # Pause before reset
            obs, info = env.reset()
            total_reward = 0
            current_direction = env.direction # Reset direction tracking

    env.close()
    print("--- Human Play Ended ---")


def train_agent(timesteps, algo, n_envs, load_path=None, save_name=DEFAULT_MODEL_NAME):
    """Trains a reinforcement learning agent."""
    print(f"\n--- Training Mode ({algo.upper()}) ---")
    print(RL_EXPLANATION)
    print(f"Training for {timesteps} timesteps...")
    print(f"Using {n_envs} parallel environments.")
    if load_path:
        print(f"Continuing training from: {load_path}")
    else:
        print("Starting training from scratch.")

    # Create vectorized environments for parallel training
    # Monitor wrapper records stats like episode reward and length
    vec_env = make_vec_env(lambda: Monitor(SnakeEnv(render_mode=None, grid_size=GRID_SIZE)),
                           n_envs=n_envs,
                           vec_env_cls=SubprocVecEnv if n_envs > 1 else DummyVecEnv)


    # --- Select Algorithm ---
    if algo.lower() == 'ppo':
        # PPO is generally a good starting point, robust and performs well
        # Tunable hyperparameters: learning_rate, n_steps, batch_size, gamma, gae_lambda, etc.
        model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log=LOG_DIR)
    elif algo.lower() == 'dqn':
        # DQN is suitable for discrete action spaces, can be sample efficient but sometimes less stable than PPO
        # Tunable hyperparameters: learning_rate, buffer_size, learning_starts, batch_size, tau, gamma, exploration_fraction, etc.
        model = DQN("MlpPolicy", vec_env, verbose=1, tensorboard_log=LOG_DIR, buffer_size=100000) # Larger buffer might be needed
    elif algo.lower() == 'a2c':
        # A2C is simpler than PPO, often faster per update but might need more samples overall
        model = A2C("MlpPolicy", vec_env, verbose=1, tensorboard_log=LOG_DIR)
    else:
        raise ValueError(f"Unsupported algorithm: {algo}. Choose 'ppo', 'dqn', or 'a2c'.")

    # Load existing model if path is provided
    if load_path:
        try:
            model = model.__class__.load(load_path, env=vec_env, tensorboard_log=LOG_DIR)
             # Reset timesteps? Optional, depends if you want total count or session count
            # model.num_timesteps = 0
            print(f"Successfully loaded model from {load_path}")
        except Exception as e:
            print(f"Error loading model from {load_path}: {e}")
            print("Starting training from scratch instead.")


    # --- Callbacks ---
    # Save checkpoints periodically
    checkpoint_callback = CheckpointCallback(save_freq=max(10000 // n_envs, 1000), # Save every N steps across all envs
                                             save_path=os.path.join(MODEL_DIR, f"{save_name}_checkpoints"),
                                             name_prefix="rl_model")

    # Evaluate the model periodically on a separate environment
    eval_env = Monitor(SnakeEnv(render_mode=None, grid_size=GRID_SIZE)) # Single env for evaluation
    eval_callback = EvalCallback(eval_env, best_model_save_path=os.path.join(MODEL_DIR, f"{save_name}_best"),
                                 log_path=os.path.join(LOG_DIR, f"{save_name}_eval"),
                                 eval_freq=max(5000 // n_envs, 500),
                                 deterministic=True, render=False)

    # --- Start Training ---
    try:
        # The learn method handles the training loop (environment interaction, policy updates)
        model.learn(total_timesteps=timesteps,
                    callback=[checkpoint_callback, eval_callback],
                    progress_bar=True) # Show progress bar
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")

    # --- Save Final Model ---
    final_model_path = os.path.join(MODEL_DIR, f"{save_name}_final.zip")
    model.save(final_model_path)
    print(f"--- Training Finished ---")
    print(f"Final model saved to: {final_model_path}")
    print(f"Best evaluation model saved in: {os.path.join(MODEL_DIR, f'{save_name}_best')}")
    print(f"TensorBoard logs saved in: {LOG_DIR}")
    print("\nTo view TensorBoard logs, run:")
    print(f"  tensorboard --logdir {LOG_DIR}")

    vec_env.close() # Close the environments


def play_agent(model_path, speed):
    """Loads a trained agent and runs the game visually."""
    print("\n--- Agent Play Mode ---")
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Please train an agent first using --mode train")
        return

    print(f"Loading model from: {model_path}")
    print(f"Agent play speed (render FPS): {speed}")

    # Determine algorithm from filename if possible (simple heuristic)
    algo = 'ppo' # Default
    if 'dqn' in model_path.lower(): algo = 'dqn'
    elif 'a2c' in model_path.lower(): algo = 'a2c'

    # Load the trained agent
    if algo == 'ppo':
        model = PPO.load(model_path)
    elif algo == 'dqn':
        model = DQN.load(model_path)
    elif algo == 'a2c':
        model = A2C.load(model_path)
    else: # Should not happen if loading worked, but just in case
        print(f"Warning: Could not determine algorithm from path, assuming PPO.")
        model = PPO.load(model_path)

    # Create a single environment for the agent to play in, with rendering
    env = SnakeEnv(render_mode="human", grid_size=GRID_SIZE)
    env.metadata['render_fps'] = speed # Control agent's visual speed

    obs, info = env.reset()
    terminated = False
    truncated = False
    total_reward = 0
    episodes = 0

    print("Starting agent gameplay... Press Ctrl+C in terminal to stop.")

    try:
        while episodes < 100: # Play a few episodes or until interrupted
            # Predict the action using the loaded model's policy
            # deterministic=True makes the agent always choose the best action (no exploration)
            action, _states = model.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, info = env.step(int(action)) # Action needs to be int
            total_reward += reward

            # Check for manual quit event in Pygame window
            for event in pygame.event.get():
                 if event.type == pygame.QUIT:
                     raise KeyboardInterrupt # Treat window close as interruption


            if terminated or truncated:
                episodes += 1
                print(f"Episode {episodes} Finished. Score: {info['score']}, Total Reward: {total_reward:.2f}")
                total_reward = 0
                # Optional: Add a pause between episodes
                time.sleep(1.5)
                if episodes < 100: # Avoid reset if loop condition met
                    obs, info = env.reset()
                else:
                    break # Exit loop after 100 episodes

    except KeyboardInterrupt:
        print("\nAgent gameplay stopped by user.")
    finally:
        env.close()
        print("--- Agent Play Ended ---")


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RL Snake Game with Human, Agent Play, and Training modes.")
    parser.add_argument("--mode", type=str, required=True, choices=["human", "train", "agent-play"],
                        help="Select the mode to run: 'human', 'train', or 'agent-play'.")
    parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "dqn", "a2c"],
                        help="RL algorithm to use for training/loading (default: ppo).")
    parser.add_argument("--timesteps", type=int, default=100000,
                        help="Total number of timesteps for training (default: 100,000).")
    parser.add_argument("--load", type=str, default=None,
                        help="Path to a pre-trained model file (.zip) to load for 'agent-play' or continue 'train'.")
    parser.add_argument("--save-name", type=str, default=None,
                        help="Custom base name for saving models during training (e.g., 'my_snake_model'). Defaults based on algo.")
    parser.add_argument("--speed", type=int, default=10,
                        help="Rendering speed (FPS) for 'human' and 'agent-play' modes (default: 10).")
    parser.add_argument("--envs", type=int, default=4,
                        help="Number of parallel environments for training (default: 4). Use 1 for no parallelism.")


    args = parser.parse_args()

    # Determine model path and save name
    model_algo_name = args.save_name if args.save_name else f"{args.algo}_snake"
    model_path_arg = args.load
    # If loading wasn't specified for agent-play, try loading the default best model
    if args.mode == "agent-play" and not model_path_arg:
        model_path_arg = os.path.join(MODEL_DIR, f"{model_algo_name}_best", "best_model.zip")
        if not os.path.exists(model_path_arg):
             model_path_arg = os.path.join(MODEL_DIR, f"{model_algo_name}_final.zip") # Fallback to final


    # Set render speed globally for the env if needed (though primarily controlled in mode functions)
    INITIAL_SPEED = args.speed
    if 'metadata' in SnakeEnv.__dict__: # Check if class attr exists
        SnakeEnv.metadata['render_fps'] = args.speed


    # --- Run selected mode ---
    if args.mode == "human":
        play_human()
    elif args.mode == "train":
        train_agent(timesteps=args.timesteps,
                    algo=args.algo,
                    n_envs=args.envs,
                    load_path=args.load, # Pass the explicit load path if given
                    save_name=model_algo_name)
    elif args.mode == "agent-play":
        if not model_path_arg or not os.path.exists(model_path_arg):
             print(f"Error: Could not find a model to load.")
             print(f"Tried default paths based on algo '{args.algo}':")
             print(f" - {os.path.join(MODEL_DIR, f'{model_algo_name}_best', 'best_model.zip')}")
             print(f" - {os.path.join(MODEL_DIR, f'{model_algo_name}_final.zip')}")
             print(f"Please specify a model using --load or train one first.")
        else:
            play_agent(model_path=model_path_arg, speed=args.speed)
    else:
        print(f"Error: Invalid mode '{args.mode}'.")