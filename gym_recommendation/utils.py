import requests
import os
import zipfile
import pandas as pd
from datetime import datetime as dt


DATA_HEADER = "user id | item id | rating | timestamp"
ITEM_HEADER = "movie id | movie title | release date | video release date | IMDb URL | " \
              "unknown | Action | Adventure | Animation | Children's | Comedy | Crime | " \
              "Documentary | Drama | Fantasy | Film-Noir | Horror | Musical | Mystery | " \
              "Romance | Sci-Fi | Thriller | War | Western"
USER_HEADER = "user id | age | gender | occupation | zip code"


# Static file path for saving and importing data set
# `gym_recommendation/data/...`
CWD = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')


def download_data():
    """
    Helper function to download MovieLens 100k data set and save to the `ml-100k`
    directory within the `/data` folder.
    :return: (void)
    """
    start_time = dt.now()
    print("Starting data download. Saving to {}".format(CWD))

    if not os.path.exists(CWD + '/ml-100k'):
        url = 'http://files.grouplens.org/datasets/movielens/ml-100k.zip'
        r = requests.get(url)

        if r.status_code != 200:
            print('Error: could not download ml100k')

        zip_file_path = os.path.join(CWD, 'ml-100k.zip')
        with open(zip_file_path, 'wb') as f:
            f.write(r.content)

        fzip = zipfile.ZipFile(zip_file_path, 'r')
        fzip.extractall(path=CWD)
        fzip.close()

        elapsed = (dt.now() - start_time).seconds
        print('Download completed in {} seconds.'.format(elapsed))


def convert_header_to_camel_case(headers):
    """Take headers available in ML 100k doc and convert it to a list of strings

    Example:
      convert "user id | item id | rating | timestamp"
      to ['user_id', 'item_id', 'rating', 'timestamp']
    """
    return headers.replace(' ', '_').split('_|_')


def import_data():
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


def import_data_for_env():
    """
    Helper function to download and import MovieLens 100k data set into Panda DataFrames.

    Function first checks if the data is already downloaded in the
    `gym_recommendation/data/` directory, and if not, downloads the data set.

    :return: Three DataFrames:
        (1) Movie rating data
        (2) Movie reference data
        (3) User reference data
    """
    if not os.path.exists(os.path.join(CWD, 'ml-100k')):
        print("Unable to find ml-100k data set in {}".format(CWD))
        download_data()
    data, item, user = import_data()
    kwargs = {
        'data': data,
        'item': item,
        'user': user
    }
    return kwargs
