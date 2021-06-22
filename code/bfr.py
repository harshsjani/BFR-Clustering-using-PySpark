import csv
import json
import math
import os
import random
import sys
import time
import copy

from collections import defaultdict, Counter
from pyspark import SparkContext


class HCluster:    
    def __init__(self, num_clusters, num_seeds, num_iterations):
        self.num_clusters = num_clusters
        self.max_iterations = num_iterations
        self.num_seeds = num_seeds

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
                min_dist = Utils.INF
                if not point in used_points:
                    for x in range(current_k):
                        dist = Utils.euclidean_distance(
                            point, centroids[x])
                        if dist < min_dist:
                            min_dist = dist
                if min_dist > maxdist:
                    maxdist = min_dist
                    new_centroid = point

            centroids[i] = new_centroid
            used_points.add(new_centroid)
        return centroids

    @staticmethod
    def get_inertia(centers, points, labels):
        # Sum of squares distance from points to centers
        inertia = 0

        for idx, p in enumerate(points):
            inertia += pow(Utils.euclidean_distance(p, centers[labels[idx]]), 2)
        return inertia

    @staticmethod
    def vector_add(A1, A2, num_dims):
        for i in range(num_dims):
            A1[i] += A2[i]

    @staticmethod
    def update_centroids(centers, centers_new, updates, cluster_counts):
        num_dims = len(centers[0])
        centers_new = [None] * len(centers)

        for cidx in range(len(centers)):
            if cluster_counts[cidx] == 0:
                centers_new[cidx] = centers[cidx]
                continue
            dims = [0] * num_dims
            for dimidx in range(num_dims):
                dims[dimidx] = updates[cidx][dimidx] / cluster_counts[cidx]
            centers_new[cidx] = dims
        return centers_new

    def single_iteration(self, points, centers, centers_new):
        labels = [None] * len(points)
        num_dims = len(points[0])
        
        # Updates stores the SUM of points along each dimension
        updates = defaultdict(lambda: [0] * num_dims)
        cluster_counts = defaultdict(int)

        # print("Centers: {}".format(centers))

        for pidx, point in enumerate(points):
            min_dist = Utils.INF

            for idx, centroid in enumerate(centers):
                dist = Utils.euclidean_distance(point, centroid)

                if dist < min_dist:
                    min_dist = dist
                    labels[pidx] = idx
            temp = copy.copy(updates[labels[pidx]])
            HCluster.vector_add(updates[labels[pidx]], point, num_dims)
            cluster_counts[labels[pidx]] += 1
        centers_new = HCluster.update_centroids(centers, centers_new, updates, cluster_counts)
        return centers, centers_new, labels
        

    def run(self, points, centers):
        centers_new = [None] * len(centers)

        for i in range(self.max_iterations):
            print("Running iteration #: {}".format(i + 1))
            centers_new, centers, labels = self.single_iteration(points, centers, centers_new)
        inertia = HCluster.get_inertia(centers_new, points, labels)

        return centers, labels, inertia

    def fit(self, points):
        best_inertia = None

        for _ in range(self.num_seeds):
            centers = self.get_centroids(points, self.num_clusters)

            centers, labels, inertia = self.run(points, centers)

            if best_inertia is None or inertia < best_inertia:
                best_labels = labels
                best_centers = centers
                best_inertia = inertia

        self.cluster_centers = best_centers
        self.labels = best_labels
        self.inertia = best_inertia

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

    def init_sets(self, sc, file_path, num_clusters):
        points = self.load_points(sc, file_path)
        points_sample = points[:27585]# random.Random().sample(points, 0.1)
        
        clusters = HCluster(num_clusters, num_seeds=2, num_iterations=5)
        clusters.fit(points_sample)
        print("Inertia: {}".format(clusters.inertia))
        # print("Cluster centers: {}".format(clusters.cluster_centers))
        print("Cluster counts: {}".format(Counter(clusters.labels)))

        # TODO: move clusters with single points or "very few" to RS
        # rs_points = points - RS.points()

        # remaining_points = points - set(rs_points)

        

    def run(self):
        sc = SparkContext.getOrCreate()

        files = os.listdir(self.input_path)
        self.init_sets(sc, files[0], self.num_clusters)

        # for idx, file_path in enumerate(files[1:]):
        #     points = self.load_points(sc, file_path)
        #     kmeans = KMeans(points, self.num_clusters)
        #     kmeans.print_cluster_data()


if __name__ == "__main__":
    st = time.time()

    runner = Runner()
    runner.run()

    print("Duration: {}".format(time.time() - st))
