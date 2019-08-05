from gym.envs.registration import register
from gym_recommendation.envs.reco_env import RecoEnv
from gym_recommendation.utils import import_data_for_env, evaluate


register(
    id=RecoEnv.id,
    entry_point='gym_recommendation.envs:RecoEnv',
    max_episode_steps=1000000,
    nondeterministic=False
)
