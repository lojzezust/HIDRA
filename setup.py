import os
from setuptools import setup

def parse_requirements(file):
    return [line for line in open(os.path.join(os.path.dirname(__file__), file))]

setup(
    name='hidra',
    python_requires='>=3.6',
    version='0.1',
    description='HIDRA Sea Level Forecasting',
    author='Lojze Žust',
    author_email='lojze.zust@gmail.com',
    packages=['hidra'],
    install_requires=parse_requirements('requirements.txt')
)
