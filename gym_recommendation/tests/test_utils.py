from gym_recommendation.utils import download_data, import_data


def test_download_data():
    download_data()


def test_import_data():
    data, item, user = import_data()


if __name__ == "__main__":
    test_download_data()
    # test_import_data()
