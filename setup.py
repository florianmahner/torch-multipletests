try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup, find_packages

setup(
    name='torch_multipletest',
    version='0.0.1',
    author='Florian P. Mahner',
    author_email='florian.mahner@gmail.com',
    license='LICENSE',
    long_description=open('README.txt').read(),
    packages=find_packages(),
)