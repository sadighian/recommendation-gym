import os
import zipfile
from datetime import datetime as dt
from typing import Dict, List, Tuple

import pandas as pd
import requests
from stable_baselines.common.base_class import ActorCriticRLModel

from . import RecoEnv

DATA_HEADER = "user id | item id | rating | timestamp"
ITEM_HEADER = "movie id | movie title | release date | video release date | IMDb URL | " \
              "unknown | Action | Adventure | Animation | Children's | Comedy | Crime | " \
              "Documentary | Drama | Fantasy | Film-Noir | Horror | Musical | Mystery | " \
              "Romance | Sci-Fi | Thriller | War | Western"
USER_HEADER = "user id | age | gender | occupation | zip code"

# Static file path for saving and importing data set
# `gym_recommendation/data/...`
CWD = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')


def download_data() -> None:
    """
    Helper function to download MovieLens 100k data set and save to the `ml-100k`
    directory within the `/data` folder.
    """
    start_time = dt.now()
    print("Starting data download. Saving to {}".format(CWD))

    if not os.path.exists(CWD):
        print('download_data() --> Making ./data/* directory...')
        os.mkdir(CWD)

    if not os.path.exists(os.path.join(CWD, 'ml-100k')):
        print('download_data() --> Making ./data/ml-100k/* directory...')
        os.mkdir(os.path.join(CWD, 'ml-100k'))

        url = 'http://files.grouplens.org/datasets/movielens/ml-100k.zip'
        r = requests.get(url)

        if r.status_code != 200:
            print('download_data() --> Error: could not download ml100k')

        zip_file_path = os.path.join(CWD, 'ml-100k.zip')
        with open(zip_file_path, 'wb') as f:
            f.write(r.content)

        with zipfile.ZipFile(zip_file_path, 'r') as f_zip:
            f_zip.extractall(path=CWD)

        elapsed = (dt.now() - start_time).seconds
        print('download_data() --> completed in {} seconds.'.format(elapsed))
    else:
        print('Using cached data located at {}.'.format(os.path.join(CWD, 'ml-100k')))


def convert_header_to_camel_case(headers: str) -> List[str]:
    """Take headers available in ML 100k doc and convert it to a list of strings

    Example:
      convert "user id | item id | rating | timestamp"
      to ['user_id', 'item_id', 'rating', 'timestamp']
    """
    return headers.replace(' ', '_').split('_|_')


def import_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Helper function to import MovieLens 100k data set into Panda DataFrames.

    :return: Three DataFrames:
        (1) Movie rating data
        (2) Movie reference data
        (3) User reference data
    """
    data = pd.read_csv(
        os.path.join(CWD, 'ml-100k', 'u.data'),
        delimiter='\t',
        names=convert_header_to_camel_case(DATA_HEADER),
        encoding='latin-1'
    )

    item = pd.read_csv(
        os.path.join(CWD, 'ml-100k', 'u.item'),
        delimiter='|',
        names=convert_header_to_camel_case(ITEM_HEADER),
        encoding='latin-1'
    )

    user = pd.read_csv(
        os.path.join(CWD, 'ml-100k', 'u.user'),
        delimiter='|',
        names=convert_header_to_camel_case(USER_HEADER),
        encoding='latin-1'
    )
    return data, item, user


def import_data_for_env() -> Dict[str, pd.DataFrame]:
    """
    Helper function to download and import MovieLens 100k data set into Panda DataFrames.

    Function first checks if the data is already downloaded in the
    `gym_recommendation/data/` directory, and if not, downloads the data set.

    :return: Three DataFrames:
        (1) Movie rating data
        (2) Movie reference data
        (3) User reference data
    """
    download_data()
    kwargs = dict([(label, data) for label, data
                   in zip(['data', 'item', 'user'], import_data())])
    return kwargs


def evaluate(model: ActorCriticRLModel, env: RecoEnv, num_steps: int = 1000) -> None:
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
            print(f"**************EPISODE #{episode_number}****************")
            print(f"Total steps = {step_count} | steps/second = {step_count / elapsed}")
            print(f"Total correct predictions = {env.total_correct_predictions}")
            print(f"Prediction accuracy = {env.total_correct_predictions / step_count}")
            obs = env.reset()
            step_count = 0
            episode_number += 1
            start_time = dt.now()
