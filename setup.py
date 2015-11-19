import os
from setuptools import setup

setup(
    name='iceview',
    version='0.0.1',
    package_data={"": ['*.jpg', '*.png', '*.json', '*.txt']},
    author='Johanna Hansen',
    author_email='jh1736@gmail.com',
    description='Tools for creating mosaics for ice imagery collected by UAVs',
    long_description=open(os.path.join(os.path.dirname(__file__),
                                       'README.md')).read(),
    license='BSD 3-clause',
    url='http://github.com/johannah/iceview',
    scripts=['scripts/build_mosaic.py', 'scripts/patchmaker.py'],
    install_requires=['numpy',
                      'scipy',
                      'scikit-image',
                      'matplotlib'],
    keywords = ['ice','uav','mosaic'],
    classifiers=['Development Status :: 3 - Alpha',
                 'Intended Audience :: Science/Research',
                 'License :: OSI Approved :: BSD License',
                 'Operating System :: OS Independent',
                 'Topic :: Scientific/Engineering'],
)
