from setuptools import setup
import os


cwd = os.path.dirname(os.path.realpath(__file__))
with open('{}/requirements.txt'.format(cwd)) as f:
      dependencies = list(map(lambda x: x.replace("\n", ""),
                              f.readlines()))


setup(
    name='reco-gym',
    version='1.0',
    url='https://github.com/RedBanies3ofThem/recommendation-gym',
    license='Apache 2.0',
    author='Jonathan Sadighian',
    author_email='jonathan.m.sadighian@gmail.com',
    description='POMDP recommendation system framework for MovieLens dataset',
    install_requires=dependencies,
    packages=['gym_recommendation'],
)
