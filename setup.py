from setuptools import setup
import os


cwd = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'requirements.txt')
with open(cwd) as f:
    dependencies = list(map(lambda x: x.replace("\n", ""), f.readlines()))


setup(
    name='recommendation-gym',
    version='1.1',
    url='https://github.com/RedBanies3ofThem/recommendation-gym',
    license='Apache 2.0',
    author='Jonathan Sadighian',
    author_email='jonathan.m.sadighian@gmail.com',
    description='POMDP recommendation system framework for MovieLens data set',
    install_requires=dependencies,
    packages=['gym_recommendation'],
)
