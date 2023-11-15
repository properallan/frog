from setuptools import setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='frog',
    #version='0.0.1',
    #description='PYthon Quasi One Dimensional Euler solver',
    author='Allan Moreira de Carvalho',
    author_email='properallan@gmail.com',
    packages=['frog'],
    install_requires= requirements,
    entry_points={ 'console_scripts': ['frog = frog.__main__:app' ] }
)