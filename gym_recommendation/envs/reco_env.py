from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from gym import Env
from gym import spaces


class RecoEnv(Env):
    # Environment static properties
    metadata = {'render.modes': ['human', 'logger']}
    id = 'reco-v0'
    actions = np.eye(5)

    def __init__(self,
                 data: pd.DataFrame,
                 item: pd.DataFrame,
                 user: pd.DataFrame,
                 seed: int = 1):
        """
        Parameterized constructor
        """
        # data for creating features
        self.data = data
        self.item = item
        self.user = user
        # features derived from data
        self.movie_genre = self._get_movie_genre(item=self.item)
        self.user_info = self._get_user_data(user=self.user)
        self.occupations = self.user.occupation.unique().tolist()
        self.num_of_occupations = len(self.occupations)
        self.user_mean = self.data.groupby('user_id').mean().to_dict()['rating']
        self.movie_mean = self.data.groupby('item_id').mean().to_dict()['rating']
        # MDP variables
        self.reward = 0.0
        self.done = False
        self.observation = None
        self.action = 0
        # other environment variables
        self.local_step_number = 0
        self._seed = seed
        self._random_state = np.random.RandomState(seed=self._seed)
        self.max_step = self.data.shape[0] - 2
        self.total_correct_predictions = 0
        # convert data to numpy for faster training
        self.data = self.data.values
        # other openAI.gym specific variables
        self.action_space = spaces.Discrete(len(RecoEnv.actions))
        self.observation_space = spaces.Box(low=-1., high=5.0,
                                            shape=self._get_observation(
                                                step_number=0).shape,
                                            dtype=np.float32)

    def step(self, action: int = 0) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Agent steps through environment
        """
        if self.done:
            self.observation = self.reset()
            return self.observation, self.reward, self.done, {}
        self.action = action
        self.reward = self._get_reward(action=action, step_number=self.local_step_number)
        self.observation = self._get_observation(step_number=self.local_step_number)
        if self.reward > 0.:
            self.total_correct_predictions += 1
        if self.local_step_number >= self.max_step:
            self.done = True
        self.local_step_number += 1
        return self.observation, self.reward, self.done, {}

    def reset(self) -> np.ndarray:
        """
        Reset the environment to an initial state
        """
        self.local_step_number = 0
        self.reward = 0.0
        self.done = False
        print(f"Reco is being reset() --> "
              f"first step = {self.local_step_number} | "
              f"Total_correct = {self.total_correct_predictions}")
        self.total_correct_predictions = 0
        return self._get_observation(step_number=self.local_step_number)

    def render(self, mode: str = 'human') -> None:
        """
        Render environment
        """
        if mode == 'logger':
            print(f"Env observation at step {self.local_step_number} is \n{self.observation}")

    def close(self) -> None:
        """
        Clear resources when shutting down environment
        """
        self.data = None
        self.user = None
        self.item = None
        print("RecoGym is being closed.")

    def seed(self, seed: int = 1) -> List[int]:
        """
        Set random seed
        """
        self._random_state = np.random.RandomState(seed=seed)
        self._seed = seed
        return [seed]

    def __str__(self) -> str:
        return f'GymID={RecoEnv.id} | seed={self._seed}'

    @staticmethod
    def _one_hot(num: int, selection: int) -> np.ndarray:
        """
        Create one-hot features
        """
        return np.eye(num, dtype=np.float32)[selection]

    @staticmethod
    def _get_movie_genre(item: pd.DataFrame) -> Dict[int, np.ndarray]:
        """
        Extract one-hot of movie genre type from dataset
        """
        movie_genre = dict([(movie_id, np.empty(19, dtype=np.float32))
                            for movie_id in item['movie_id'].tolist()])
        for movie_id in range(1, len(movie_genre)):
            movie_genre[movie_id] = item.iloc[movie_id, 5:].values.astype(np.float32)
        return movie_genre

    @staticmethod
    def _get_user_data(user: pd.DataFrame) -> Dict[int, Dict[str, Union[int, str]]]:
        """
        Create dictionary of user stats (e.g., age, occupation, gender)
        to use as inputs into other functions.
        """
        tmp_user = user.drop(['zip_code'], axis=1)
        tmp_user.index = tmp_user.user_id
        tmp_user = tmp_user.drop(['user_id'], axis=1)
        return tmp_user.to_dict(orient='index')

    def _get_movie_genre_buckets(self, movie_id: int = 1) -> np.ndarray:
        """
        Extract one-hot of movie genre type for a specific movie_id
        """
        return self.movie_genre.get(movie_id, np.empty(19, dtype=np.float32))

    def _get_age_buckets(self, age: int = 10) -> np.ndarray:
        """
        Extract one-hot of age group for a specific age
        """
        if age < 10:
            bucket_number = 0
        elif age < 20:
            bucket_number = 1
        elif age < 30:
            bucket_number = 2
        elif age < 40:
            bucket_number = 3
        elif age < 50:
            bucket_number = 4
        elif age < 60:
            bucket_number = 5
        else:
            bucket_number = 6
        return self._one_hot(num=7, selection=bucket_number)

    def _get_occupation_buckets(self, job: str = 'none') -> np.ndarray:
        """
        Extract one-hot of occupation type for a specific job
        """
        selection = self.occupations.index(job)
        return self._one_hot(num=self.num_of_occupations, selection=selection)

    def _get_gender_buckets(self, gender: str = 'm') -> np.ndarray:
        """
        Extract one-hot of gender type for a specific gender (e.g., M or F)
        """
        sex = gender.upper()
        sex_id = 0 if sex == 'M' else 1
        return self._one_hot(num=2, selection=sex_id)

    def _get_observation(self, step_number: int = 0) -> np.ndarray:
        """
        Get features and concatenate them into one observation

        Features=
          user_mean:
            Average rating given by a specific user_id
          movie_mean:
            Average rating for a specific movie_id
          movie_genre_bucket:
            One-hot of the movie type
          age_bucket:
            One-hot of user's age range
          occupation_bucket:
            One-hot of the user's job
          gender_bucket:
            One-hot of the user's gender (only M or F)
        """
        # lookup keys
        user_id = self.data[step_number, 0]
        movie_id = self.data[step_number, 1]
        # values for one_hot
        user_age = self.user_info[user_id]['age']
        user_occupation = self.user_info[user_id]['occupation']
        user_gender = self.user_info[user_id]['gender']
        # features
        user_mean = np.array([self.user_mean.get(user_id, 3.) / 5.], dtype=np.float32)
        movie_mean = np.array([self.movie_mean.get(movie_id, 3.) / 5.], dtype=np.float32)
        movie_genre_bucket = self._get_movie_genre_buckets(movie_id=movie_id)
        age_bucket = self._get_age_buckets(age=user_age)
        occupation_bucket = self._get_occupation_buckets(job=user_occupation)
        gender_bucket = self._get_gender_buckets(gender=user_gender)
        # concatenate it all together
        return np.concatenate((user_mean, movie_mean, movie_genre_bucket,
                               age_bucket, occupation_bucket, gender_bucket))

    def _get_reward(self, action: int, step_number: int) -> float:
        """
        Calculate reward for a given state and action
        """
        users_rating = int(self.data[step_number, 2])
        predicted_rating = int(action) + 1  # to compensate for zero-index
        prediction_difference = abs(predicted_rating - users_rating)
        reward = 0.
        if prediction_difference == 0:
            reward += 1.
        else:
            reward += np.log(float(prediction_difference) / 5.)  # Added log-penalty
        return reward
