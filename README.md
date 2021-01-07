# PythonVRFT Library - Version 0.0.3
VRFT Adaptive Control Library written in Python. Aim of this library is to provide an implementation of the VRFT (Virtual Reference Feedback Tuning) algorithm.

_Author_: Alessio Russo (PhD Student at KTH - alesssior@kth.se)

![alt tag](https://github.com/rssalessio/PythonVRFT/blob/master/examples/1_example.png)
## License
Our code is released under the GPLv3 license (refer to the [LICENSE](https://github.com/rssalessio/PythonVRFT/blob/master/LICENSE) file for details).

## Requirements
- Python 3.9.1
- NumPy 1.19.5
- SciPy 1.6.0

## Installation
Check the requirements, but the following command should install all the packages.
Run the following command from  root folder:
```sh
pip install . 
``` 

## Examples
Examples are located in the examples/ folder. At the moment only 1 example is available.

## Tests
To execute tests run the following command
```sh
python -m unittest
``` 

## Changelog
- [**DONE - V0.0.2**][26.03.2017] Implement the basic VRFT algorithm (1 DOF. offline, linear controller, controller expressed as scalar product theta*f(z))
- [**DONE - V0.0.3**][05.01.2020] Code refactoring and conversion to Python 3; Removed support for Python Control library.
- [**In Progress**][07.01.2020-] Add Documentation and Latex formulas
- [**TODO**] Add MIMO Support
- [**TODO**] Add IV Support
- [**TODO**] Generalize to other kind of controllers (e.g., neural nets)
