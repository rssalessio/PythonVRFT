# PythonVRFT Library - Version 0.0.5
VRFT Adaptive Control Library written in Python. Aim of this library is to provide an implementation of the VRFT (Virtual Reference Feedback Tuning) algorithm.

You can find the package also at the following [link](https://pypi.org/project/pythonvrft/)

_Author_: Alessio Russo (PhD Student at KTH - alesssior@kth.se)


![alt tag](https://github.com/rssalessio/PythonVRFT/blob/master/examples/example2.png)
## License
Our code is released under the GPLv3 license (refer to the [LICENSE](https://github.com/rssalessio/PythonVRFT/blob/master/LICENSE) file for details).

## Requirements
To run the library you need atleast Python 3.5.

Other dependencies:
- NumPy (1.19.5)
- SciPy (1.6.0)

## Installation
- Install last stable version ```sh pip install pythonvrft```
- Install from source: ```sh
pip install . 
``` 

## Examples
Examples are located in the examples/ folder. At the moment there are examples available. 
Check example3 to see usage of instrumental variables.

## Tests
To execute tests run the following command
```sh
python -m unittest
``` 

## Changelog
- [**V. 0.0.2**][26.03.2017] Implement the basic VRFT algorithm (1 DOF. offline, linear controller, controller expressed as scalar product theta*f(z))
- [**V. 0.0.3**][05.01.2020] Code refactoring and conversion to Python 3; Removed support for Python Control library.
- [**V. 0.0.5**][08.01.2020] Add Instrumental Variables (IVs) Support
- [**In Progress**][07.01.2020-] Add Documentation and Latex formulas
- [**TODO**] Add MIMO Support
- [**TODO**] Generalize to other kind of controllers (e.g., neural nets)
- [**TODO**] Add Cython support

## Citations
If you find this code useful in your research, please, consider citing it:
>@misc{pythonvrft,
>  author       = {Alessio Russo},
>  title        = {Python VRFT Library},
>  year         = 2020,
>  doi          = {},
>  url          = { https://github.com/rssalessio/PythonVRFT }
>}

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
