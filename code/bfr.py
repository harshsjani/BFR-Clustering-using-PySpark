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
    def get_centroids(pwi, num_clusters):
        centroids = [None] * num_clusters
        used_points = set()
        N = len(pwi)
        init_idx = random.randint(0, N)
        centroids[0] = pwi[init_idx][1]
        used_points.add(init_idx)

        for i in range(1, num_clusters):
            maxdist = 0
            new_centroid = None
            used_idx = None

            for idx, point in pwi:
                if not idx in used_points:
                    min_dist = Utils.INF
                    for x in range(i):
                        dist = Utils.euclidean_distance(
                            point, centroids[x])
                        if dist < min_dist:
                            min_dist = dist
                if min_dist > maxdist:
                    maxdist = min_dist
                    new_centroid = point
                    used_idx = idx

            centroids[i] = new_centroid
            used_points.add(used_idx)
        return centroids

    @staticmethod
    def get_inertia(centers, points, labels):
        # Sum of squares distance from points to centers
        inertia = 0

        for idx, p in points:
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

    def single_iteration(self, pwi, centers, centers_new):
        labels = {}
        num_dims = len(pwi[0][1])
        
        # Updates stores the SUM of points along each dimension
        updates = defaultdict(lambda: [0] * num_dims)
        cluster_counts = defaultdict(int)

        for pidx, point in pwi:
            min_dist = Utils.INF

            for idx, centroid in enumerate(centers):
                dist = Utils.euclidean_distance(point, centroid)

                if dist < min_dist:
                    min_dist = dist
                    labels[pidx] = idx
            HCluster.vector_add(updates[labels[pidx]], point, num_dims)
            cluster_counts[labels[pidx]] += 1
        centers_new = HCluster.update_centroids(centers, centers_new, updates, cluster_counts)
        return centers, centers_new, labels
        
    def run(self, pwi, centers):
        centers_new = [None] * len(centers)

        for i in range(self.max_iterations):
            print("Running iteration #: {}".format(i + 1))
            centers_new, centers, labels = self.single_iteration(pwi, centers, centers_new)
        inertia = 0#HCluster.get_inertia(centers_new, pwi, labels)

        return centers, labels, inertia

    def fit(self, pwi):
        best_inertia = None

        for _ in range(self.num_seeds):
            st = time.time()
            centers = self.get_centroids(pwi, self.num_clusters)
            print("Time to get centroids: {}".format(time.time() - st))

            centers, labels, inertia = self.run(pwi, centers)

            if best_inertia is None or inertia < best_inertia:
                best_labels = labels
                best_centers = centers
                best_inertia = inertia

        self.cluster_centers = best_centers
        self.labels = best_labels
        self.inertia = best_inertia


class Utils:
    INF = float('inf')

    @staticmethod
    def euclidean_distance(p1, p2):
        # vectors in n-dimensional space
        num_dims = len(p1)
        sqsm = 0

        for i in range(num_dims):
            sqsm += pow((p1[i] - p2[i]), 2)
        return math.sqrt(sqsm)

    @staticmethod
    def mahalanobis_distance(p, c, std):
        N = len(p)
        dist = 0
        for i in range(N):
            dist += pow((p[i] - c[i]) / std[i], 2)
        return math.sqrt(dist)


class DiscardSet:
    def __init__(self, points, center) -> None:
        self.num_points = len(points)
        self.num_dims = len(points[0])
        self.dsums = [0] * self.num_dims
        self.dsqsums = [0] * self.num_dims
        
        for p in points:
            for i in range(self.num_dims):
                self.dsums[i] += p[i]
                self.dsqsums[i] += pow(p[i], 2)
        self.center = center

    def get_stds(self):
        stds = [0 for _ in range(self.num_dims)]
        for i in range(self.num_dims):
            stds[i] = math.sqrt(self.dsqsums[i] / self.num_points - pow(self.dsums[i] / self.num_points, 2))
        return stds


class Runner:
    def __init__(self) -> None:
        self.input_path = sys.argv[1]
        self.num_clusters = int(sys.argv[2])
        self.cluster_out = sys.argv[3]
        self.intermediate_out = sys.argv[4]
        self.intermediate_header = ["round_id", "nof_cluster_discard", "nof_point_discard",
                                    "nof_cluster_compression", "nof_point_compression", "nof_point_retained"]
        self.discard_sets = []
        self.compressed_sets = []
        self.retained_sets = []
        self.out_dict = {}
    
    def load_points(self, sc, file_path):
        return sc.textFile(os.path.join(self.input_path, file_path)).map(lambda row: row.split(",")).map(lambda row: (str(row[0]), list(map(lambda x: float(x), row[1:]))))

    def init_RS(self, points, labels, cluster_ctr):
        singleton_clusters = set()
        RS = []
        for k, v in cluster_ctr.least_common():
            if v == 1:
                singleton_clusters.add(k)
        for idx, label in enumerate(labels):
            if label in singleton_clusters:
                pass

    def init_DSs(self, pwi, labels, centers):
        N = len(pwi)
        ds_points = [[] for _ in range(self.num_clusters)]

        for i in range(N):
            pidx = pwi[i][0]
            point = pwi[i][1]
            label = labels[pidx]
            ds_points[label].append(point)
        
        for i in range(self.num_clusters):
            self.discard_sets.append(DiscardSet(ds_points[i], centers[i]))

    def init_sets(self, pointsRDD):
        pwi = pointsRDD.collect()
        num_points = len(pwi)
        print("Total number of points: {}".format(num_points))
        fraction = 0.2
        sample_idx = math.ceil(num_points * fraction)
        print("Num points sampled: {}".format(sample_idx + 1))
        points_sample = pwi[:sample_idx]
        
        clusters = HCluster(self.num_clusters * 3, num_seeds=1, num_iterations=5)
        clusters.fit(points_sample)
        
        ctr = Counter(clusters.labels.values())

        outlier_labels = set()

        for k, v in ctr.items():
            if v == 1:
                outlier_labels.add(k)
        
        print("Outlier labels: {}".format(outlier_labels))

        inlier_points = []
        outlier_points = []

        for point_with_idx in points_sample:
            idx = point_with_idx[0]

            if clusters.labels[idx] not in outlier_labels:
                inlier_points.append(point_with_idx)
            else:
                outlier_points.append(point_with_idx)
        
        inlier_points.extend(pwi[sample_idx:])

        print("Outlier points: {}".format(outlier_points))

        print("Number of inlier points + rest: {}".format(len(inlier_points)))
        clustering = HCluster(self.num_clusters, num_seeds=1, num_iterations=5)
        clustering.fit(inlier_points)

        print("Number of label keys: {}".format(len(clustering.labels)))
        print("Label keys: {}".format(list(clustering.labels.keys())[:1000]))
        self.init_DSs(pwi, clustering.labels, clustering.cluster_centers)

        # Write initial assignment
        for idx, _ in inlier_points:
            label = clustering.labels[idx]
            self.out_dict[idx] = label

    @staticmethod
    def assign_to_ss(point, summarized_sets, alpha):
        choice = None
        smallest_dist = Utils.INF

        for i in range(len(summarized_sets)):
            ss = summarized_sets[i]
            dist = Utils.mahalanobis_distance(point, ss.center, ss.get_stds())

            if dist < alpha * math.sqrt(ss.num_dims) and dist < smallest_dist:
                smallest_dist = dist
                choice = i
        return choice

    def run(self):
        sc = SparkContext.getOrCreate()
        sc.setLogLevel("OFF")

        files = list(sorted(os.listdir(self.input_path)))
        print("Files: {}".format(files))

        for idx, file_path in enumerate(files):
            print("Processing index: {}/{}".format(idx + 1, len(files)))
            pointsRDD = self.load_points(sc, file_path)
            
            # For the first round, we want to do some extra stuff
            if idx == 0:
                self.init_sets(pointsRDD)
            else:
                dss = self.discard_sets
                assigned = pointsRDD.map(lambda idx_point: (idx_point[0], Runner.assign_to_ss(idx_point[1], dss, alpha=2)))
                rem_points = assigned.filter(lambda row: row[1] == None)
                ds_assigned = assigned.filter(lambda row: row[1] != None)
                self.out_dict.update(ds_assigned.collectAsMap())
                rem_minus_one = rem_points.mapValues(lambda val: -1)
                self.out_dict.update(rem_minus_one.collectAsMap())
            print("Number of assigned points in out dict: {}".format(len(self.out_dict)))
        
        with open(self.cluster_out, "w+") as f:
            json.dump(self.out_dict, f)
        with open(self.intermediate_out, "w+") as f:
            f.write(",".join(self.intermediate_header))



if __name__ == "__main__":
    st = time.time()

    runner = Runner()
    runner.run()

    print("Duration: {}".format(time.time() - st))
