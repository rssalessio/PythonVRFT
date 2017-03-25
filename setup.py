from setuptools import setup

setup(name = 'vrft',
	version = '0.0.1',
	description = 'VRFT Python Library',
	long_description = 'VRFT Python Library',
	keywords = 'VRFT Virtual Reference Feedback Tuning Adaptive Control',
	url = '',
	author = 'Alessio Russo',
	author_email = 'alessio.russo@rapyuta-robotics.com',
	license='GPL3',
	packages=['vrft'],
	zip_safe=False,
	install_requires = [
		'scipy',
		'numpy',
		'control',
	],
	test_suite = 'nose.collector',
	test_require = ['nose']
)