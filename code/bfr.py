import csv
import json
import math
import os
import random
import sys
import time

from collections import defaultdict
from pyspark import SparkContext


class Utils:
    INF = 10 ** 12

    @staticmethod
    def euclidean_distance(p1, p2):
        # vectors in n-dimensional space
        num_dims = len(p1)
        sqsm = 0

        for i in range(num_dims):
            sqsm += pow((p1[i] - p2[i]), 2)
        return math.sqrt(sqsm)
    
    @staticmethod
    def get_centroids(points, num_clusters):
        centroids = [None] * num_clusters
        used_points = set()
        centroids[0] = random.Random().choice(points)
        used_points.add(centroids[0])
        current_k = 1

        for i in range(1, num_clusters):
            maxdist = 0
            new_centroid = None

            for point in points:
                distance_sum = 0
                if not point in used_points:
                    for x in range(current_k):
                        distance_sum += Utils.euclidean_distance(
                            point, centroids[x])
                if distance_sum > maxdist:
                    maxdist = distance_sum
                    new_centroid = point

            centroids[i] = new_centroid
            used_points.add(new_centroid)
        return centroids


class KMeans:
    def __init__(self, points, k: int) -> None:
        self.points = points
        self.num_clusters = k
        self.num_dims = len(points[0])
        self.clusters = defaultdict(set)
        self.run()


    """
    Assigns cluster IDs to each point
    """

    def run(self) -> None:
        centroids = Utils.get_centroids(self.points, self.num_clusters)
        centroid_id_map = {centroid: idx for idx,
                           centroid in enumerate(centroids)}
        id_centroid_map = {idx: centroid for idx,
                           centroid in enumerate(centroids)}

        for point in self.points:
            min_dist = Utils.INF
            centroid_id = None

            for centroid in centroids:
                dist = Utils.euclidean_distance(point, centroid)

                if dist < min_dist:
                    min_dist = dist
                    centroid_id = centroid_id_map[centroid]
            self.clusters[centroid_id].add(point)

    def print_cluster_data(self):
        print("Number of clusters: {}".format(len(self.clusters)))
        print("Number of points per cluster:")
        for k in self.clusters:
            print("{}: {}".format(k, len(self.clusters[k])))


class Runner:
    def __init__(self) -> None:
        self.input_path = sys.argv[1]
        self.num_clusters = int(sys.argv[2])
        self.cluster_out = sys.argv[3]
        self.intermediate_out = sys.argv[4]
        self.intermediate_header = ["round_id", "nof_cluster_discard", "nof_point_discard",
                                    "nof_cluster_compression", "nof_point_compression", "nof_point_retained"]

    def load_points(self, sc, file_path):
        return sc.textFile(os.path.join(self.input_path, file_path)).map(lambda row: list(map(lambda x: float(x), row.split(",")[1:]))).map(lambda row: tuple(row)).collect()

    def init_sets(self, sc, file_path):
        points = self.load_points(sc, file_path)
        points_sample = random.Random().sample(points, 0.1)
        kmeans_result = KMeans(points_sample, self.num_clusters * 5)
        

    def run(self):
        sc = SparkContext.getOrCreate()

        files = os.listdir(self.input_path)
        for idx, file_path in enumerate(files[1:]):
            points = self.load_points(sc, file_path)
            kmeans = KMeans(points, self.num_clusters)
            kmeans.print_cluster_data()


if __name__ == "__main__":
    st = time.time()

    runner = Runner()
    runner.run()

    print("Duration: {}".format(time.time() - st))
