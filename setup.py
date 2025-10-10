from setuptools import find_packages, setup

# Read dependencies from requirements.txt
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='loan-approval',
    version='0.0.1',
    author='abhi',
    author_email='abhigilbile74@gmail.com',
    packages=find_packages(),
    install_requires=requirements,
)
