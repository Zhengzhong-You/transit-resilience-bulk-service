import unittest

from evaluate_frozen_route_phase import normalized_hash_paths


class ReproductionPathsTest(unittest.TestCase):
    def test_frozen_windows_manifest_can_be_verified_on_a_posix_machine(self):
        windows={r'runs\calibration\kernel.npz':'abc123'}
        posix={'runs/calibration/kernel.npz':'abc123'}
        self.assertEqual(normalized_hash_paths(windows),normalized_hash_paths(posix))


if __name__=='__main__':unittest.main()
