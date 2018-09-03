from setuptools import find_packages, setup

DESC = 'A python library for manipulating sequential and-inverter gates.'

setup(
    name='py-aiger-bv',
    version='0.2.1',
    description=DESC,
    url='http://github.com/mvcisback/py-aiger-bv',
    author='Marcell Vazquez-Chanlatte',
    author_email='marcell.vc@eecs.berkeley.edu',
    license='MIT',
    install_requires=[
        'funcy',
        'py-aiger',
        'attr'
    ],
    packages=find_packages(),
)
