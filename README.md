# PythonVRFT Library
VRFT Adaptive Control Library written in Python.

**Author**: Alessio Russo (alessior.wordpress.com - russo.alessio@outlook.com)

Aim of this library is to provide an implementation of the VRFT (Virtual Reference Feedback Tuning) algorithm.

![alt tag](https://github.com/rssalessio/PythonVRFT/blob/master/examples/1_example.png)

Installation
------
Run the following command from  root folder:
```sh
pip install . 
``` 
Dependencies: numpy, scipy, control

Tests
------
To execute tests run the following command
```sh
python setup.py test
``` 

Examples
------
Examples are located in the examples/ folder. At the moment only 1 example is available.

Objectives
------
- [**DONE**, 26.03.2017] Implement the basic VRFT algorithm (1 DOF. offline, linear controller, controller expressed as scalar product theta*f(z))
- [**TODO**] Implement  multiple DOF, sensitivity analysis
- [**TODO**] Implement online version
- [**TODO**] Generalize to other kind of controllers
- [**TODO**] Advanced work (non linear systems ?)



