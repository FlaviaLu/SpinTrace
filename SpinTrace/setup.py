from setuptools import setup, find_packages
from os import path

# read the contents of README.rst
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='spintrace',
    packages=find_packages(),
    version='1.0.0',
    license='MIT',
    description='A Python library to study rotational properties of small bodies using survey-based photometry',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Flavia L. Rommel',
    author_email='flavialuanerommel@gmail.com',
    url='https://github.com/FlaviaLu/SpinTrace',
    keywords=['science', 'astronomy', 'small bodies','photometry'],
    install_requires=[
        'numpy>=2.3.1',
        'astropy>=7.1.0',
        'astroquery>=0.4.10',
        'matplotlib>=3.10.3',
        'scipy>=1.16.0',
        'matplotlib>=3.10.3',
        'plotly>=5.24.1',
        'space-rocks>=1.9.13',
        'kaleido>=1.0.0'
    ],
    python_requires=">=3.13.5",
)