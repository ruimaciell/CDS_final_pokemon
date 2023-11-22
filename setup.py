try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup

import library_hw4

def get_requirements(requirements_path='requirements.txt'):
    with open(requirements_path) as fp:
        return [x.strip() for x in fp.read().split('\n') if not x.startswith('#')]

get_requirements()

setup(
    name='library_hw4',
    version=library_hw4.__version__,
    description='Second library',
    author='Mikel, Luis, Rui',
    packages=find_packages(where='', exclude=['tests']),
    install_requires=get_requirements(),
    setup_requires=['pytest-runner', 'wheel'],
    url='https://github.com/lalvarezpoli/DS_HW3.git',
    classifiers=[
        'Programming Language :: Python >= 3.7.16'
    ]
)
