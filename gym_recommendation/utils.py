import requests
import os
import zipfile
import pandas as pd


DATA_HEADER = "user id | item id | rating | timestamp"
ITEM_HEADER = "movie id | movie title | release date | video release date | IMDb URL | " \
              "unknown | Action | Adventure | Animation | Children's | Comedy | Crime | " \
              "Documentary | Drama | Fantasy | Film-Noir | Horror | Musical | Mystery | " \
              "Romance | Sci-Fi | Thriller | War | Western"
USER_HEADER = "user id | age | gender | occupation | zip code"

CWD = os.path.dirname(os.path.realpath(__file__))


def download_data():
    if not os.path.exists(CWD + '/ml-100k'):
        url = 'http://files.grouplens.org/datasets/movielens/ml-100k.zip'
        r = requests.get(url)

        if r.status_code != 200:
            print('Error: could not download ml100k')

        with open('ml-100k.zip', 'wb') as f:
            f.write(r.content)
        fzip = zipfile.ZipFile('ml-100k.zip', 'r')
        fzip.extractall('.')
        fzip.close()
        print('ml100k downloaded')


def convert_header_to_camel_case(headers):
    """Take headers available in ML 100k doc and convert it to a list of strings

    Example:
      convert "user id | item id | rating | timestamp"
      to ['user_id', 'item_id', 'rating', 'timestamp']
    """
    return headers.replace(' ', '_').split('_|_')


def import_data():
    data = pd.read_csv(
        CWD + '/ml-100k/u.data',
        delimiter='\t',
        names=convert_header_to_camel_case(DATA_HEADER),
        encoding='latin-1'
    )

    item = pd.read_csv(
        CWD + '/ml-100k/u.item',
        delimiter='|',
        names=convert_header_to_camel_case(ITEM_HEADER),
        encoding='latin-1'
    )

    user = pd.read_csv(
        CWD + '/ml-100k/u.user',
        delimiter='|',
        names=convert_header_to_camel_case(USER_HEADER),
        encoding='latin-1'
    )
    return data, item, user
