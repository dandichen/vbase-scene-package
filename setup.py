from setuptools import setup, find_packages

setup(
    name='scene-tag',
    version='0.0.1',
    author='Chenyang Li',
    author_email='lichenyangsun1@gmail.com',
    description='vbase-scene',
    classifiers=['Private :: Do Not Upload'],
    requires=['mxnet, opencv, numpy, scipy, sklearn'],
    url='https://github.com/TuSimple/vbase-scene',
    packages=find_packages(exclude=["tests"])
)