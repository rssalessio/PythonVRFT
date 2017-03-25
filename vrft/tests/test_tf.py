from unittest import TestCase
from vrft.utilities.tf import *
import numpy as np

class TestTF(TestCase):
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

		with self.assertRaises(ValueError):
			systemOrder(np.array(10), np.array(1))

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
