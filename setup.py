from setuptools import setup, find_packages
from os import path

# read the contents of README.rst
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='ZTFrc',
    packages=find_packages(),
    version='0.1',
    license='MIT',
    description='Python library to serach for rotational periodicity in ZTF public data',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Flavia L. Rommel',
    author_email='flavialuanerommel@gmail.com',
    url='https://github.com/FlaviaLu/ZTFrc',
    keywords=['science', 'astronomy', 'small bodies'],
    install_requires=[
        'numpy>=2.2.0',
        'pyerfa>=2.0.1.5',
        'astropy>=7.0.0',
        'astroquery>=0.4.7',
        'matplotlib>=3.9.3',
        'scipy>=1.14.1',
        'Pyedra>=0.3.1',
    ],
    python_requires=">=3.13.1",
)