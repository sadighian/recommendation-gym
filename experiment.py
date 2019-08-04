import gym
import gym_recommendation
from datetime import datetime as dt

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import PPO1, PPO2

import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--learning_rate',
                    default=float(3e-4),
                    help="Learning rate for optimizer",
                    type=float)
parser.add_argument('--n_steps',
                    default=64,
                    help="Number of steps for PPO to roll out",
                    type=int)
parser.add_argument('--tensorboard_log',
                    default="./ppo2_recommendations",
                    help="Logging directory for TensorBoard log",
                    type=str)
parser.add_argument('--save_model',
                    default=True,
                    help="Save model after training (True = yes | False = no)",
                    type=bool)
parser.add_argument('--training_steps',
                    default=int(1e6),
                    help="Number of steps to performing training",
                    type=int)
parser.add_argument('--evaluation_steps',
                    default=int(1e5),
                    help="Number of steps to performing on evaluation",
                    type=int)
parser.add_argument('--seed',
                    default=1,
                    help="Random number seed for evaluation",
                    type=int)
parser.add_argument('--num_of_layers',
                    default=2,
                    help="Number of hidden layers in the MLP",
                    type=int)
parser.add_argument('--num_of_neurons',
                    default=64,
                    help="Number of neurons in each hidden layer",
                    type=int)
parser.add_argument('--nminibatches',
                    default=4,
                    help="Number of mini-batches in PPO",
                    type=int)
kwargs = vars(parser.parse_args())


def evaluate(model, env, num_steps=1000):
    """
    Evaluate a RL agent
    """
    start_time = dt.now()
    obs = env.reset()
    step_count = 0
    episode_number = 1
    for i in range(num_steps):
        step_count += 1
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        if done:
            elapsed = (dt.now() - start_time).seconds
            print("**************EPISODE #{}****************".format(episode_number))
            print("Total steps=", step_count, "steps/second=", step_count / elapsed)
            print("Total correct predictions=", env.total_correct_predictions)
            print("Prediction accuracy=", env.total_correct_predictions / step_count)
            obs = env.reset()
            step_count = 0
            episode_number += 1
            start_time = dt.now()


def main(kwargs):
    start_time = dt.now()
    number_of_cpu = os.cpu_count()
    envs = SubprocVecEnv([lambda: gym.make(gym_recommendation.RecoEnv.id,
                                           **gym_recommendation.import_data_for_env())
                          for _ in range(number_of_cpu)])
    ppo_configs = {
        'learning_rate': kwargs['learning_rate'],  # lambda f: f * float(3e-4),
        'n_steps': kwargs['n_steps'],
        'tensorboard_log': kwargs['tensorboard_log'],
        'nminibatches': min(number_of_cpu, kwargs['nminibatches']),
        'noptepochs': number_of_cpu,
        'policy_kwargs': dict(net_arch=[kwargs['num_of_neurons']]*kwargs['num_of_layers']),
        'gamma': 0.0,
        'ent_coef': 0.01,
        'vf_coef': 0.5,
        'max_grad_norm': 0.5,
        'lam': 0.95,
        'cliprange': 0.2,
        'verbose': 0,
        '_init_setup_model': True,
    }
    print('*********************************')
    print("ppo_configs:", ppo_configs)
    print('*********************************')

    model = PPO2(MlpPolicy, envs, **ppo_configs)
    model.learn(total_timesteps=kwargs['training_steps'], seed=kwargs['seed'])
    elapsed = (dt.now() - start_time).seconds
    print("Finished training in {} seconds".format(elapsed))

    if kwargs['save_model']:
        save_name = 'PPO2_Recommendations'
        print('Saving PPO model as {}'.format(save_name))
        model.save(save_name)

    evaluate(model=model,
             env=gym.make(gym_recommendation.RecoEnv.id,
                          **gym_recommendation.import_data_for_env()),
             num_steps=kwargs['evaluation_steps'])
    elapsed = (dt.now() - start_time).seconds
    print("Finished training AND evaluation in {} seconds".format(elapsed))


if __name__ == "__main__":
    print("Starting experiment with PPO2 at {}".format(dt.now()))
    main(kwargs=kwargs)
