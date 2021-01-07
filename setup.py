from setuptools import setup

setup(name = 'pythonvrft-alessior',
	version = '0.0.3',
	description = 'VRFT Python Library',
	long_description = 'VRFT Python Library',
	keywords = 'VRFT Virtual Reference Feedback Tuning Adaptive Control',
	url = 'https://github.com/rssalessio/PythonVRFT/',
	author = 'Alessio Russo',
	author_email = 'alessior@kth.se',
	license='GPL3',
	packages=['vrft', 'test'],
	zip_safe=False,
	install_requires = [
		'scipy',
		'numpy',
	],
	test_suite = 'nose.collector',
	test_requires = ['nose'],
	classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GPL3 License",
        "Operating System :: OS Independent",
    ],
	python_requires='>=3.5',
)
