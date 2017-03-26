# PythonVRFT
VRFT Adaptive Control Library
Author: Alessio Russo (alessior.wordpress.com - russo.alessio@outlook.com)

Aim of this library is to provide an implementation of the VRFT (Virtual Reference Feedback Tuning) algorithm.


Objectives

- [DONE] Implement the basic VRFT algorithm (1 DOF. offline, linear controller, controller expressed as scalar product theta*f(z))
- [TODO] Implement  multiple DOF, sensitivity analysis
- [TODO] Implement online version
- [TODO] Generalize to other kind of controllers
- [TODO] Advanced work (non linear systems ?)


How to install:

pip install . from this folder

To run tests use python setup.py test

![alt tag](https://github.com/rssalessio/PythonVRFT/blob/master/examples/1_example.png)

