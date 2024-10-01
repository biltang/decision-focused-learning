from setuptools import setup, find_packages

setup(
    name='decision_focused_learning',
    version='0.1',
    packages=find_packages(include=['decision_learning', 'decision_learning.*']),
)