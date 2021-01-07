# iddata.py - iddata object definition
# Analogous to the iddata object in Matlab sysid
#
# Code author: [Alessio Russo - alessior@kth.se]
# Last update: 07th January 2020, by alessior@kth.se
#
# Copyright [2017-2020] [Alessio Russo - alessior@kth.se]  
# This file is part of PythonVRFT.
# PythonVRFT is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
# PythonVRFT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with PythonVRFT.  If not, see <http://www.gnu.org/licenses/>.
#


import numpy as np

class iddata(object):
    """
     iddata is a class analogous to the iddata object in Matlab
     It is used to save input/output data.
    """

    def __init__(self, y: np.ndarray = None,
                 u: np.ndarray = None,
                 ts: float = None,
                 y0: np.ndarray = None):
        """
        Input/output data (suppors SISO systems only)
        Parameters
        ----------
        y: np.ndarray
            Output data
        u: np.ndarray
            Input data
        ts: float
            sampling time
        y0: np.ndarray
            Initial conditions
        """
        self.y = np.array(y) if not isinstance(y, np.ndarray) else np.array([y]).flatten()
        self.u = np.array(u) if not isinstance(u, np.ndarray) else np.array([u]).flatten()
        self.ts = float(ts)
        self.y0 = np.array(y0) if not isinstance(y0, np.ndarray) else np.array([y0]).flatten()


        if self.y0.ndim == 0:
            self.y0 = np.expand_dims(self.y0, axis=0)


    def checkData(self):
        """ Checks validity of the data """
        if (self.y.shape != self.u.shape):
            raise ValueError("Input and output size do not match.")

        if (np.isclose(self.ts, 0.0) == True):
            raise ValueError("Sampling time can not be zero.")

        if (self.ts < 0.0):
            raise ValueError("Sampling time can not be negative.")

        if (self.y0 is None):
            return True

        return True

    def copy(self):
        """ Returns a copy of the object """
        return iddata(self.y, self.u, self.ts, self.y0)
