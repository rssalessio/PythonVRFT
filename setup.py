from setuptools import setup

setup(name = 'vrft',
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
	test_requires = ['nose']
)
