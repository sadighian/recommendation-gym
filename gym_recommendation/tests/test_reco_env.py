import numpy as np
from datetime import datetime as dt
import gym
from gym_recommendation import RecoEnv, import_data_for_env


def test_recommendation_environment():
    """
    Test case to validate RecoEnv is compatible with OpenAI Gym's Make() function.
    :return: (void)
    """
    start_time = dt.now()

    env = gym.make(RecoEnv.id, **import_data_for_env())

    i = 0
    env.reset()
    total_rewards = 0.

    while True:
        i += 1

        if i % 5000 == 0:
            action = np.random.randint(env.action_space.n)
            env.render(mode='logger')  # render observation steps
        else:
            action = 0

        state, reward, done, _ = env.step(action)
        total_rewards += reward

        if done:
            elapsed = (dt.now() - start_time).seconds
            print("Simulation completed on step " +
                  "{} in {} seconds at a rate of {:.2f} steps/sec".format(
                      i, elapsed, i/elapsed))
            break
    print('total_correct_predictions: {}'.format(env.env.total_correct_predictions))
    print('total_rewards: {}'.format(total_rewards))
