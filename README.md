# PythonVRFT Library - Version 0.0.3
VRFT Adaptive Control Library written in Python.

**Author**: Alessio Russo (PhD Student at KTH - alesssior@kth.se)

Aim of this library is to provide an implementation of the VRFT (Virtual Reference Feedback Tuning) algorithm.

![alt tag](https://github.com/rssalessio/PythonVRFT/blob/master/examples/1_example.png)

Installation
------
Run the following command from  root folder:
```sh
pip install . 
``` 
Dependencies: numpy, scipy

Tests
------
To execute tests run the following command
```sh
python -m unittest
``` 

Examples
------
Examples are located in the examples/ folder. At the moment only 1 example is available.

Objectives
------
- [**DONE - V0.0.2**][26.03.2017] Implement the basic VRFT algorithm (1 DOF. offline, linear controller, controller expressed as scalar product theta*f(z))
- [**DONE - V0.0.3**][05.01.2020] Code refactoring and conversion to Python 3; Removed support for Python Control library.
- [**TODO**] Add Documentation and Latex formulas
- [**TODO**] Add MIMO Support
- [**TODO**] Add IV Support
- [**TODO**] Generalize to other kind of controllers (e.g., neural nets)
