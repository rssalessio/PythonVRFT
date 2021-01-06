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
# Code author: [Alessio Russo - alessior@kth.se]
# Last update: 06th January 2020, by alessior@kth.se
#

import numpy as np

class iddata(object):
    def __init__(self, y: np.ndarray = None,
                 u: np.ndarray = None,
                 ts: float = None,
                 y0: np.ndarray = None):
        self.y = np.array(y) if not isinstance(y, np.ndarray) else np.array([y]).flatten()
        self.u = np.array(u) if not isinstance(u, np.ndarray) else np.array([u]).flatten()
        self.ts = float(ts)
        self.y0 = np.array(y0) if not isinstance(y0, np.ndarray) else np.array([y0]).flatten()


        if self.y0.ndim == 0:
            self.y0 = np.expand_dims(self.y0, axis=0)


    def checkData(self):
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
        return iddata(self.y, self.u, self.ts, self.y0)
