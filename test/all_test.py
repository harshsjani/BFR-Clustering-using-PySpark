import unittest
import sys


class Test_KMeans(unittest.TestCase):
    def test_single_point(self):
        sys.path.append("F:/Projects/DM_HW/hw4/code")
        import bfr
        points = [[(1, 2)]]
        kmeans = bfr.KMeans(points, 1)
        kmeans.run()
        print(kmeans.get_data())


if __name__ == "__main__":
    unittest.main()
