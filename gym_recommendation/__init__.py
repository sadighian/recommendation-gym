from gym.envs.registration import register
from gym_recommendation.envs.reco_env import RecoEnv


register(
    id=RecoEnv.id,
    entry_point='gym_recommendation.envs:RecoEnv',
    max_episode_steps=1000000,
    nondeterministic=False
)
