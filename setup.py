from setuptools import setup
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setup(name = 'pythonvrft',
    version = '0.0.6',
    description = 'VRFT Python Library',
    long_description = long_description,
    long_description_content_type = 'text/markdown',
    keywords = ['VRFT', 'Virtual Reference Feedback Tuning',
                'Data Driven Control', 'Adaptive Control'],
    url = 'https://github.com/rssalessio/PythonVRFT/',
    author = 'Alessio Russo',
    author_email = 'alessior@kth.se',
    license = 'GPL3',
    packages = ['vrft', 'test'],
    zip_safe = False,
    install_requires = [
        'scipy',
        'numpy',
    ],
    test_suite = 'nose.collector',
    test_requires = ['nose'],
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Topic :: Scientific/Engineering"
    ],
    python_requires = '>=3.5',
)
