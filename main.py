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
from snake_env import SnakeEnv, Direction, GRID_SIZE, INITIAL_SPEED # Make sure this imports the updated SnakeEnv

# --- Configuration ---
MODEL_DIR = "models"
LOG_DIR = "logs"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Default model filename base
DEFAULT_MODEL_NAME_BASE = "snake" # Algo prefix will be added

# --- RL Explanation ---
RL_EXPLANATION = """
=================================
Reinforcement Learning Concepts
=================================

1. Environment (SnakeEnv):
   - Represents the game world (the Snake grid).
   - Defines:
     - State/Observation Space: What the agent 'sees'. Now a Dictionary Space:
       * 'mlp_features': Vector of danger, direction, food/tail location.
       * 'cnn_features': Small image grid around the head for local spatial awareness.
     - Action Space: What the agent can 'do' (Up, Down, Left, Right).
     - Reward Signal: Feedback guiding the learning (+10 food, -10 death, +/- distance, step penalty).
     - Dynamics: How the state changes based on an action.

2. Agent (The Trained Model):
   - Learner and decision-maker aiming to maximize cumulative reward.
   - Contains a Policy and optionally a Value Function.

3. Policy ("MultiInputPolicy"):
   - The agent's strategy mapping observation (Dict) to action.
   - Uses a CombinedExtractor (automatically selected by SB3):
     * CNN processes the 'cnn_features' (local grid image).
     * MLP processes the 'mlp_features' (vector).
     * Outputs are combined and fed to final layers to select action probabilities/Q-values.
   - Training adjusts weights in both CNN and MLP parts.

4. Reward Signal (Defined in SnakeEnv.step):
   - Crucial for learning. Guides the agent towards desired behaviors (eating, surviving, efficiency).
   - Reward Shaping (distance rewards) helps speed up learning.

5. Learning Process (e.g., PPO Algorithm):
   - Agent interacts with the environment, collects experiences (state, action, reward, next_state).
   - Algorithm uses experiences to update the MultiInputPolicy network.
   - Balances Exploration (trying new things) and Exploitation (using known good actions).

6. Backtracking / Credit Assignment:
   - Algorithms use techniques (Value Functions, Advantage Estimation) to link rewards/penalties
     back to the actions that caused them, enabling learning of long-term consequences.
=================================
"""

# --- Mode Functions ---

def play_human():
    """Runs the Snake game in human-playable mode."""
    print("\n--- Human Play Mode ---")
    print("Controls: Arrow Keys (Up, Down, Left, Right)")
    print("Press 'Q' to Quit.")

    # Use render_mode="human" for direct display
    # Make sure it uses the updated SnakeEnv class
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
            action_val = env.direction.value
        else:
            # Update current_direction based on input, env handles invalid reversals
            current_direction = Direction(action_val)


        # Step with the chosen action value
        # Observation is now a dictionary, but human play doesn't use it directly
        obs, reward, terminated, truncated, info = env.step(action_val)
        total_reward += reward

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            print("Resetting game...")
            time.sleep(2) # Pause before reset
            obs, info = env.reset()
            total_reward = 0
            current_direction = env.direction # Reset direction tracking

    env.close()
    print("--- Human Play Ended ---")


def train_agent(timesteps, algo, n_envs, load_path=None, save_name=DEFAULT_MODEL_NAME_BASE):
    """Trains a reinforcement learning agent using MultiInputPolicy."""
    print(f"\n--- Training Mode ({algo.upper()}) with MultiInputPolicy ---")
    print(RL_EXPLANATION)
    print(f"Training for {timesteps} timesteps...")
    print(f"Using {n_envs} parallel environments.")
    if load_path:
        print(f"Continuing training from: {load_path}")
    else:
        print("Starting training from scratch.")

    # Use the save_name base provided, prefixed by algorithm
    full_save_name = f"{algo}_{save_name}"

    # Create vectorized environments for parallel training
    # Monitor wrapper records stats like episode reward and length
    # Ensure lambda uses the updated SnakeEnv
    vec_env = make_vec_env(lambda: Monitor(SnakeEnv(render_mode=None, grid_size=GRID_SIZE)),
                           n_envs=n_envs,
                           vec_env_cls=SubprocVecEnv if n_envs > 1 else DummyVecEnv)


    # --- Select Algorithm ---
    # Use "MultiInputPolicy" for Dict observation spaces
    policy_type = "MultiInputPolicy"

    if algo.lower() == 'ppo':
        model = PPO(policy_type, vec_env, verbose=1, tensorboard_log=LOG_DIR,
                    # Consider tuning hyperparameters for MultiInputPolicy
                    # learning_rate=0.0003, n_steps=2048, batch_size=64, gamma=0.99, ...
                   )
    elif algo.lower() == 'dqn':
        # DQN might need careful tuning with MultiInputPolicy
        model = DQN(policy_type, vec_env, verbose=1, tensorboard_log=LOG_DIR,
                    buffer_size=100000, # May need larger buffer
                    learning_starts=5000, # Allow more initial exploration
                    # exploration_final_eps=0.05, exploration_fraction=0.2, ...
                   )
    elif algo.lower() == 'a2c':
        model = A2C(policy_type, vec_env, verbose=1, tensorboard_log=LOG_DIR)
    else:
        raise ValueError(f"Unsupported algorithm: {algo}. Choose 'ppo', 'dqn', or 'a2c'.")

    # --- Loading Model ---
    # SB3 load automatically detects MultiInputPolicy from the saved model structure.
    if load_path:
        try:
            model = model.__class__.load(load_path, env=vec_env, tensorboard_log=LOG_DIR)
            print(f"Successfully loaded model from {load_path}")
        except Exception as e:
            print(f"Error loading model from {load_path}: {e}")
            print("Starting training from scratch instead.")


    # --- Callbacks ---
    checkpoint_callback = CheckpointCallback(save_freq=max(20000 // n_envs, 1000), # Save frequency
                                             save_path=os.path.join(MODEL_DIR, f"{full_save_name}_checkpoints"),
                                             name_prefix="rl_model")

    # Ensure eval_env also uses the updated SnakeEnv
    eval_env = Monitor(SnakeEnv(render_mode=None, grid_size=GRID_SIZE))
    eval_callback = EvalCallback(eval_env, best_model_save_path=os.path.join(MODEL_DIR, f"{full_save_name}_best"),
                                 log_path=os.path.join(LOG_DIR, f"{full_save_name}_eval"),
                                 eval_freq=max(10000 // n_envs, 500), # Eval frequency
                                 deterministic=True, render=False)

    # --- Start Training ---
    try:
        model.learn(total_timesteps=timesteps,
                    callback=[checkpoint_callback, eval_callback],
                    progress_bar=True,
                    tb_log_name=full_save_name) # Log under specific name
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")

    # --- Save Final Model ---
    final_model_path = os.path.join(MODEL_DIR, f"{full_save_name}_final.zip")
    model.save(final_model_path)
    print(f"--- Training Finished ---")
    print(f"Final model saved to: {final_model_path}")
    print(f"Best evaluation model saved in: {os.path.join(MODEL_DIR, f'{full_save_name}_best')}")
    print(f"TensorBoard logs saved in: {LOG_DIR}")
    print("\nTo view TensorBoard logs, run:")
    print(f"  tensorboard --logdir {LOG_DIR}")

    vec_env.close() # Close the environments


def play_agent(model_path, speed, algo):
    """Loads a trained agent (expected MultiInputPolicy) and runs the game visually."""
    print("\n--- Agent Play Mode ---")
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Please train an agent first using --mode train")
        return

    print(f"Loading model from: {model_path}")
    print(f"Agent play speed (render FPS): {speed}")

    # Load the trained agent - SB3 load infers the policy type
    if algo == 'ppo':
        model = PPO.load(model_path)
    elif algo == 'dqn':
        model = DQN.load(model_path)
    elif algo == 'a2c':
        model = A2C.load(model_path)
    else:
        print(f"Warning: Unknown algorithm '{algo}' specified. Attempting to load as PPO.")
        try:
            model = PPO.load(model_path)
        except Exception as e1:
            print(f"Failed loading as PPO: {e1}")
            print("Attempting load with DQN...")
            try:
                model = DQN.load(model_path)
            except Exception as e2:
                 print(f"Failed loading as DQN: {e2}")
                 print("Cannot load model.")
                 return


    # Create a single environment with rendering, using the updated SnakeEnv
    env = SnakeEnv(render_mode="human", grid_size=GRID_SIZE)
    env.metadata['render_fps'] = speed # Control agent's visual speed

    obs, info = env.reset()
    terminated = False
    truncated = False
    total_reward = 0
    episodes = 0

    print("Starting agent gameplay... Press Ctrl+C in terminal or close window to stop.")

    try:
        while episodes < 100: # Play a few episodes or until interrupted
            # Predict action using the loaded MultiInputPolicy model
            # Observation 'obs' is now a dictionary, model handles it automatically
            action, _states = model.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, info = env.step(int(action)) # Action needs to be int
            total_reward += reward

            # Check for manual quit event in Pygame window
            try:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        raise KeyboardInterrupt # Treat window close as interruption
            except pygame.error: # Handle cases where display might close unexpectedly
                print("Pygame display error. Stopping agent.")
                break


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
    parser = argparse.ArgumentParser(description="RL Snake Game with Human, Agent Play, and Training modes (using MultiInputPolicy).")
    parser.add_argument("--mode", type=str, required=True, choices=["human", "train", "agent-play"],
                        help="Select the mode to run: 'human', 'train', or 'agent-play'.")
    parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "dqn", "a2c"],
                        help="RL algorithm to use for training/loading (default: ppo).")
    parser.add_argument("--timesteps", type=int, default=200000, # Increased default for more complex policy
                        help="Total number of timesteps for training (default: 200,000).")
    parser.add_argument("--load", type=str, default=None,
                        help="Path to a pre-trained model file (.zip) to load for 'agent-play' or continue 'train'.")
    parser.add_argument("--save-name", type=str, default=DEFAULT_MODEL_NAME_BASE,
                        help="Custom base name for saving models during training (e.g., 'my_snake'). Algo prefix added automatically.")
    parser.add_argument("--speed", type=int, default=15, # Slightly faster default speed
                        help="Rendering speed (FPS) for 'human' and 'agent-play' modes (default: 15).")
    parser.add_argument("--envs", type=int, default=4,
                        help="Number of parallel environments for training (default: 4). Use 1 for no parallelism.")


    args = parser.parse_args()

    # Determine model path and save name
    model_algo_name = f"{args.algo}_{args.save_name}" # e.g., ppo_snake
    model_path_arg = args.load

    # If loading wasn't specified for agent-play, try loading the default best/final model
    if args.mode == "agent-play" and not model_path_arg:
        best_path = os.path.join(MODEL_DIR, f"{model_algo_name}_best", "best_model.zip")
        final_path = os.path.join(MODEL_DIR, f"{model_algo_name}_final.zip")
        if os.path.exists(best_path):
             model_path_arg = best_path
        elif os.path.exists(final_path):
             model_path_arg = final_path


    # Set render speed globally for the env if needed
    INITIAL_SPEED = args.speed
    if 'metadata' in SnakeEnv.__dict__:
        SnakeEnv.metadata['render_fps'] = args.speed


    # --- Run selected mode ---
    if args.mode == "human":
        play_human()
    elif args.mode == "train":
        train_agent(timesteps=args.timesteps,
                    algo=args.algo,
                    n_envs=args.envs,
                    load_path=args.load,
                    save_name=args.save_name) # Pass base name
    elif args.mode == "agent-play":
        if not model_path_arg or not os.path.exists(model_path_arg):
             print(f"Error: Could not find a model to load.")
             print(f"Tried default paths based on algo '{args.algo}' and name '{args.save_name}':")
             print(f" - {os.path.join(MODEL_DIR, f'{model_algo_name}_best', 'best_model.zip')}")
             print(f" - {os.path.join(MODEL_DIR, f'{model_algo_name}_final.zip')}")
             print(f"Please specify a model using --load or train one first.")
        else:
            # Pass the determined algo to play_agent for correct loading class selection
            play_agent(model_path=model_path_arg, speed=args.speed, algo=args.algo)
    else:
        print(f"Error: Invalid mode '{args.mode}'.")