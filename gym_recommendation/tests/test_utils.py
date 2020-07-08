import os

from gym_recommendation.utils import CWD, download_data, import_data, import_data_for_env


def test_download_data() -> None:
    print("test_download_data\n*****")
    download_data()
    assert os.path.exists(os.path.join(CWD, 'ml-100k')), "test_download_data() failed."


def test_import_data() -> None:
    print("test_import_data\n*****")
    try:
        data, item, user = import_data()
        print(f"data peak:\n{data.head()}")
        print(f"item peak:\n{item.head()}")
        print(f"user peak:\n{user.head()}")
    except Exception as ex:
        print("Unable to test_import_data()")
        print(f"Exception = {ex}")


def test_import_data_for_env() -> None:
    print("test_import_data_for_env\n*****")
    kwargs = import_data_for_env()
    print(f"data peak:\n{kwargs['data'].head()}")
    print(f"item peak:\n{kwargs['item'].head()}")
    print(f"user peak:\n{kwargs['user'].head()}")


if __name__ == '__main__':
    test_import_data_for_env()
