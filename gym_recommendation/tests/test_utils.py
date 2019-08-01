from gym_recommendation.utils import download_data, import_data, CWD
import os


def test_download_data():
    download_data()
    assert os.path.exists(os.path.join(CWD, 'ml-100k')), "test_download_data() failed."


def test_import_data():
    try:
        data, item, user = import_data()
        print("data peak:\n{}".format(data.head()))
        print("item peak:\n{}".format(item.head()))
        print("user peak:\n{}".format(user.head()))
    except Exception as ex:
        print("Unable to test_import_data()")
        print("Exception = {}".format(ex))


if __name__ == "__main__":
    test_download_data()
    test_import_data()
