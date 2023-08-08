from setuptools import setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='frog',
    version='0.0.0',
    description='Flow Reconstruction on GitHub',
    author='Allan Moreira de Carvalho',
    author_email='properallan@gmail.com',
    packages=['frog'],
    install_requires= requirements,
)