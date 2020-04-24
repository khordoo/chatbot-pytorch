import unittest
import torch
import numpy as np
from src.utility import Utility


class TestUtility(unittest.TestCase):
    """Tests The utility class methods"""

    def test_split_train_test_splits_based_on_fraction(self):
        """tests the methods properly splits the input sequence."""
        sources = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        targets = [11, 22, 33, 44, 55, 66, 77, 88, 99]
        test_fraction = 0.4
        train_source, train_target, test_source, test_target = Utility.split_train_test(sources, targets,
                                                                                        test_fraction=test_fraction)
        self.assertEqual(train_source, [4, 5, 6, 7, 8, 9])
        self.assertEqual(train_target, [44, 55, 66, 77, 88, 99])
        self.assertEqual(test_source, [1, 2, 3])
        self.assertEqual(test_target, [11, 22, 33])

    def test_tensorize(self):
        """tests all items of the sequence were converted to tensor"""
        tensors = Utility.tensorize([1, 2, 3])
        for tensor in tensors:
            self.assertIsInstance(tensor, torch.FloatTensor)

