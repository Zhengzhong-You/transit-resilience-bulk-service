import unittest

import numpy as np

from validate_service_revision import residual_queues


class QueueValidationTest(unittest.TestCase):
    def test_leftovers_and_initial_queue_are_conserved(self):
        arrivals = [0, 5, 0, 5]
        np.testing.assert_array_equal(residual_queues(arrivals, 3), [0, 2, 0, 2])
        np.testing.assert_array_equal(residual_queues(arrivals, 3, initial=4), [1, 3, 0, 2])


if __name__ == '__main__':
    unittest.main()
