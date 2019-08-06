from gym_recommendation.utils import download_data, import_data, import_data_for_env, CWD
import os


def test_download_data():
    print("test_download_data\n*****")
    download_data()
    assert os.path.exists(os.path.join(CWD, 'ml-100k')), "test_download_data() failed."


def test_import_data():
    print("test_import_data\n*****")
    try:
        data, item, user = import_data()
        print("data peak:\n{}".format(data.head()))
        print("item peak:\n{}".format(item.head()))
        print("user peak:\n{}".format(user.head()))
    except Exception as ex:
        print("Unable to test_import_data()")
        print("Exception = {}".format(ex))


def test_import_data_for_env():
    print("test_import_data_for_env\n*****")
    kwargs = import_data_for_env()
    print("data peak:\n{}".format(kwargs['data'].head()))
    print("item peak:\n{}".format(kwargs['item'].head()))
    print("user peak:\n{}".format(kwargs['user'].head()))


if __name__ == '__main__':
    test_import_data_for_env()
