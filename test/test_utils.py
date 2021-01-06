from unittest import TestCase
from vrft.utils import *
from vrft.extended_tf import ExtendedTF
from vrft.vrft_algo import virtualReference
import numpy as np
import scipy.signal as scipysig

class TestUtils(TestCase):
    def test_deconvolve(self):
        t_start = 0
        t_end = 10
        t_step = 1e-2
        t = np.arange(t_start, t_end, t_step)
        sys = ExtendedTF([0.5], [1, -0.9], dt=t_step)
        u = np.random.normal(size=t.size)
        _, y = scipysig.dlsim(sys, u, t)
        y = y[:, 0]
        data = iddata(y, u, t_step, [0])
        r1, _ = virtualReference(data, sys.num, sys.den)
        r2 = deconvolve_signal(sys, data.y, data.ts)
        self.assertTrue(np.linalg.norm(r2-r1[:r2.size], np.infty) <  1e-3)


    def test_checkSystem(self):
        a = [1, 0, 1]
        b = [1, 0, 2]
        self.assertTrue(checkSystem(a,b))

        b = [1, 0, 2, 4]
        self.assertTrue(checkSystem(a,b))

        a = [1]
        self.assertTrue(checkSystem(a,b))

        a = [1, 0, 1]
        b = [1,0]
        with self.assertRaises(ValueError):
            checkSystem(a,b)

        b = [1]
        with self.assertRaises(ValueError):
            checkSystem(a,b)

    def test_systemOrder(self):
        with self.assertRaises(ValueError):
            systemOrder(0, 0)

        with self.assertRaises(ValueError):
            systemOrder(1, 0)

        with self.assertRaises(ValueError):
            systemOrder([0], 0)

        with self.assertRaises(ValueError):
            systemOrder([0], [0])

        with self.assertRaises(ValueError):
            systemOrder([0, 0], [0, 0])

        with self.assertRaises(ValueError):
            systemOrder([0], [0, 0])


        self.assertEqual(systemOrder([1],[1]), (0, 0))
        self.assertEqual(systemOrder([1, 1],[1, 1]), (1,1))
        self.assertEqual(systemOrder([1, 1, 3],[1, 1]), (2,1))
        self.assertEqual(systemOrder([1, 1, 3],[1]), (2,0))
        self.assertEqual(systemOrder([1, 1],[1, 1, 1]), (1,2))
        self.assertEqual(systemOrder([1],[1, 1, 1]), (0,2))
        self.assertEqual(systemOrder([0, 1],[1, 1, 1]), (0,2))
        self.assertEqual(systemOrder([0,0,1],[1, 1, 1]), (0,2))
        self.assertEqual(systemOrder([0,0,1],[0, 1, 1, 1]), (0,2))
        self.assertEqual(systemOrder([0,0,1],[0, 0, 1, 1, 1]), (0,2))
