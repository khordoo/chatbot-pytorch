import unittest
import torch
from src.utility import Utility


class TestUtility(unittest.TestCase):
    """Tests The utility class methods"""

    def setUp(self):
        self.inputs = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    def test_split_train_test_splits_based_on_fraction(self):
        """tests the methods properly splits the input sequence."""
        test_fraction = 0.4
        train, test = Utility.split_train_test(self.inputs, test_fraction=test_fraction)
        self.assertEqual(train, [4, 5, 6, 7, 8, 9])
        self.assertEqual(test, [1, 2, 3])

    def test_tensorize(self):
        """tests all items of the sequence were converted to tensor"""
        tensors = Utility.tensorize(self.inputs)
        for tensor in tensors:
            self.assertIsInstance(tensor, torch.FloatTensor)
