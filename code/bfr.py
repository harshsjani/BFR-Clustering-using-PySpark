import csv
import json
import math
import random
import sys
import time


class Utils:
    @staticmethod
    def euclidean_distance(p1, p2):
        # vectors in n-dimensional space
        num_dims = len(p1)
        sqsm = 0

        for i in range(num_dims):
            sqsm += pow((p1[i] - p2[i]), 2)
        return math.sqrt(sqsm)

class KMeans:
    def __init__(self, data, k: int) -> None:
        # data - a list of points in n-dimensional space
        self.data = data
        self.num_clusters = k
        self.num_dims = len(data[0])

    def get_centroids(self):
        centroids = [None] * self.num_clusters
        used_points = set()
        centroids[0] = random.Random().choice(self.data)
        used_points.add(centroids[0])
        current_k = 1

        for i in range(1, self.num_clusters):
            maxdist = 0
            new_centroid = None

            for point in self.data:
                distance_sum = 0
                if not point in used_points:
                    for x in range(current_k):
                        distance_sum += Utils.euclidean_distance(point, centroids[x])
                if distance_sum > maxdist:
                    maxdist = distance_sum
                    new_centroid = point
            
            centroids[i] = new_centroid
            used_points.add(new_centroid)

    """
    Assigns cluster IDs to each point
    """
    def run(self) -> None:
        centroids = self.get_centroids()

class Runner:
    def __init__(self) -> None:
        self.input_path = sys.argv[1]
        self.num_clusters = int(sys.argv[2])
        self.cluster_out = sys.argv[3]
        self.intermediate_out = sys.argv[4]
        self.intermediate_header = ["round_id", "nof_cluster_discard", "nof_point_discard",
                                    "nof_cluster_compression", "nof_point_compression", "nof_point_retained"]


if __name__ == "__main__":
    st = time.time()

    runner = Runner()
    runner.run()

    print("Duration: {}".format(time.time() - st))
