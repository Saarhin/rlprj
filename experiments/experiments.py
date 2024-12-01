import argparse
import zipfile
import gymnasium as gym
import numpy as np
import sys
import os
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import PPO, DQN
from tqdm import tqdm


#TODO: 1. Make a virtual envrionment
#2. Have all the custom enviornment code files in the venv
#3. Have a requirements.txt file so that people can just easily install ali.


#TODO: What to do in the callback
class RewardCallback(BaseCallback):

    def __init__(self, model_id, env_id, seed, delta_t, alpha, path, task_ID, verbose=0, total_timesteps=200_000):
        super().__init__(verbose)
        self.model_id = model_id
        self.env_id = env_id
        self.seed = seed
        self.delta_t = delta_t
        self.alpha = alpha
        self.path = path
        self.task_ID = task_ID  # Add task_ID
        self.eps_returns_list = []
        self.eps_return = 0
        self.steps_rewards = []
        self.done_status = []

        self.total_timesteps = total_timesteps/delta_t
        self.progress_bar = None
        self.steps_completed = 0
    
    def _on_training_start(self) -> None:
        self.progress_bar = tqdm(total=self.total_timesteps, desc="Training Progress", unit="steps")


    def _on_step(self) -> bool:

        self.steps_rewards.append(self.locals['rewards'])
        self.done_status.append(self.locals['dones'])
        self.eps_return += self.locals['rewards']
        if self.locals['dones'] == True:
            self.eps_returns_list.append(self.eps_return)
            self.eps_return =0

        self.steps_completed += 1
        self.progress_bar.update(1)
       
        return True

    def _on_training_end(self) -> None:
        if self.progress_bar:
            self.progress_bar.close()

        filename_returns = f"Alg{self.model_id}_env{self.env_id}_seed{self.seed}_tmultiplier{self.delta_t}_alpha{self.alpha}_RETURNS.npy"
        filename_rewards = f"Alg{self.model_id}_env{self.env_id}_seed{self.seed}_tmultiplier{self.delta_t}_alpha{self.alpha}_REWARDS.npy"

        full_returns = os.path.join(self.path, filename_returns)
        full_rewards = os.path.join(self.path, filename_rewards)

        np.save(full_returns, self.eps_returns_list)
        np.save(full_rewards, np.column_stack((self.steps_rewards, self.done_status)))

        # # Create a ZIP file
        # zip_filename = os.path.join(self.path, f"{self.task_ID}.zip")
        # with zipfile.ZipFile(zip_filename, 'w') as zipf:
        #     zipf.write(full_returns, os.path.basename(full_returns))
        #     zipf.write(full_rewards, os.path.basename(full_rewards))

        # # Optionally, clean up the .npy files if no longer needed
        # os.remove(full_returns)
        # os.remove(full_rewards)

        return True
     

class Experiment:
    def __init__(self, 
                 model_id:str,
                 env_id:str, 
                 delta_t = 1,
                 gamma = 1,
                 epsilon = 0.3,
                 learning_rate = 0.001,
                 #steps_per_update = 1,
                 policy_kwargs = dict(net_arch=[64,64,64]),
                 device = 'cpu',
                 seed = 0, #TODO: Please think about this?
                #  callback = RewardCallback(),
                 total_timesteps = 200_000,
                 max_episode_steps = None, 
                 save_path = None,
                 batch_size = 16,
                 buffer_size = 500, 
                 target_update = 100, #TODO: Find out the default values of these parameters.
                 task_ID="01",  # Add task_ID parameter
                 ):
        
        self.returns = 0
        self.delta_t = delta_t #TODO: How can we compare graphs if we make the delta t as a multiple of the default time step?
        self.gamma = gamma
        # self.callback = callback
        self.save_path = save_path
        self.batch_size = batch_size
        self.epsilon = epsilon

        self.learning_rate = learning_rate
        #self.steps_per_update = steps_per_update
        self.policy_kwargs = policy_kwargs
        self.device = device
        self.seed = seed

        self.buffer_size = buffer_size
        self.target_update = target_update

        self.model_id = model_id
        self.max_episode_steps = max_episode_steps
        self.env_id = env_id
        self.total_timesteps = total_timesteps/self.delta_t
        self.task_ID = task_ID  # Store task_ID

        # Initialize the callback with required parameters
        self.callback = RewardCallback(
            model_id=self.model_id,
            env_id=self.env_id,
            seed=self.seed,
            delta_t=self.delta_t,
            alpha=self.learning_rate,
            path=self.save_path,
            task_ID=self.task_ID,  # Pass task_ID to the callback
        )

        #make the environment
        self.make_env()

        #define the model
        self.define_model()

    def make_env(self):

        #First register the environment:
        if self.env_id == "CartPole":
            self.env_id = "CustomCartPole-v0"
            gym.register(
            id="CustomCartPole-v0",
            entry_point="custom_cartpole:CustomCartPoleEnv",
            reward_threshold = 475.0,
            max_episode_steps = 500, 
            )
        elif self.env_id == "Acrobot":
            self.env_id ="CustomAcroBot-v1"
            gym.register(
            id="CustomAcroBot-v1",
            entry_point="custom_acrobot:CustomAcrobotEnv",
            reward_threshold = -100,
            max_episode_steps = 500, 
             )
        elif self.env_id == "LunarLander":
            self.env_id = "CustomAcroBot-v1" #TODO: Create CustomLunarLander
            gym.register(
            id="CustomAcroBot-v1",
            entry_point="custom_acrobot:CustomAcrobtEnv",
            reward_threshold = 200,
            max_episode_steps = 1000, 
             )

        self.env = gym.make(self.env_id, 
                        dt_multip = self.delta_t, 
                        max_episode_steps= self.max_episode_steps, 
                        )
        self.env.reset(seed = self.seed)

    def run(self):
        self.model.learn(total_timesteps =self.total_timesteps, callback = self.callback)

    def save_data(self):
        self.model.save(self.save_path) #TODO: Deal with the path after you realize how to run on compute canada.
  
    

    #Please first run make env
    def define_model(self):
        
        if self.model_id == 'PPO':
            self.model = PPO("MlpPolicy", 
                        env=self.env, 
                        verbose=0, 
                        policy_kwargs=self.policy_kwargs,
                        learning_rate=self.learning_rate,
                        seed = self.seed,
                        gamma = self.gamma,
                        device= self.device,
                        batch_size=self.batch_size,
                        # exploration_initial_eps = self.epsilon,
                        # exploration_final_eps = self.epsilon,
                        )
        elif self.model_id == 'DQN':
            self.model = DQN("MlpPolicy", 
                        env = self.env, 
                        verbose=0, 
                        policy_kwargs=self.policy_kwargs,
                        learning_rate=self.learning_rate,
                        seed = self.seed,
                        buffer_size=self.buffer_size,
                        gamma = self.gamma,
                        device=self.device, 
                        batch_size= self.batch_size,
                        target_update_interval = self.target_update,
                        exploration_initial_eps = self.epsilon,
                        exploration_final_eps = self.epsilon,
                        )


def main():
    # Step 1: Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run experiments with different configurations.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for the experiment")
    parser.add_argument("--alg", type=str, default="PPO", choices=["PPO", "DQN"], help="Algorithm to use")
    parser.add_argument("--env", type=str, default="CartPole", help="Environment ID")
    parser.add_argument("--t_multip", type=float, default=1.0, help="Time step multiplier (delta_t)")
    #parser.add_argument("--no_t_step", type=float, default=10000, help="No. of tiem steps")
    parser.add_argument("--alph", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--path", type=str, default="./", help="Path to save results")
    parser.add_argument("--task_ID", type=str, default="01", help="Task_ID")
    args = parser.parse_args()



    exp = Experiment(model_id = args.alg,
                 env_id = args.env, 
                 delta_t = args.t_multip,
                 gamma = 1,
                 epsilon = 0,
                 learning_rate = args.alph,
                 #steps_per_update = 1,
                 policy_kwargs = dict(net_arch=[64,64,64]),
                 device = 'cpu',
                 seed = args.seed, #TODO: Please think about this?
                #  callback = RewardCallback(),
                 total_timesteps = 200_000,
                 max_episode_steps = None, 
                 save_path = args.path,
                 batch_size = 16,
                 buffer_size = 500, 
                 target_update = 100, #TODO: Find out the default values of these parameters.
                 task_ID=args.task_ID,  # Pass task_ID here
                 )

    # Step 3: Run the experiment
    exp.run()

    # Step 4: Save the results
    exp.save_data()


if __name__ == "__main__":
    main()
