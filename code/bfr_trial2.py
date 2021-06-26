import csv
import json
import math
import os
import random
import sys
import time

from itertools import combinations
from collections import defaultdict, Counter


class HCluster:    
    def __init__(self, num_clusters, num_iterations):
        self.num_clusters = num_clusters
        self.max_iterations = num_iterations

    def get_centroids(self, pwi, num_clusters):
        centroids = [None] * num_clusters
        N = len(pwi)
        init_idx = random.randint(0, N)
        centroids[0] = pwi[init_idx][1]

        for i in range(1, num_clusters):
            maxdist = 0
            new_centroid = None

            for idx, point in pwi:
                min_dist = Utils.INF
                for x in range(i):
                    dist = Utils.euclidean_distance(
                        point, centroids[x])
                    if dist < min_dist:
                        min_dist = dist
                if min_dist > maxdist and min_dist != Utils.INF:
                    maxdist = min_dist
                    new_centroid = point

            centroids[i] = new_centroid
        return centroids

    def vector_add(self, A1, A2, num_dims):
        for i in range(num_dims):
            A1[i] += A2[i]

    def update_centroids(self, centers, centers_new, updates, cluster_counts):
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
            
            self.vector_add(updates[labels[pidx]], point, num_dims)
            cluster_counts[labels[pidx]] += 1
        
        centers_new = self.update_centroids(centers, centers_new, updates, cluster_counts)
        return centers, centers_new, labels
        
    def run(self, pwi, centers):
        centers_new = [None] * len(centers)

        for i in range(self.max_iterations):
            print("Running iteration #: {}".format(i + 1))
            centers_new, centers, labels = self.single_iteration(pwi, centers, centers_new)

        return centers, labels, 0

    def fit(self, pwi):
        st = time.time()
        centers = self.get_centroids(pwi, self.num_clusters)
        print("Time to get centroids: {}".format(time.time() - st))

        centers, labels, _ = self.run(pwi, centers)

        best_labels = labels
        best_centers = centers

        self.cluster_centers = best_centers
        self.labels = best_labels


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
            if std[i] != 0:
                dist += pow((p[i] - c[i]) / std[i], 2)
        return math.sqrt(dist)

    @staticmethod
    def union_stds(cs1, cs2):
        stds = []
        N = cs1.num_points + cs2.num_points

        for i in range(cs1.num_dims):
            dsums = cs1.dsums[i] + cs2.dsums[i]
            dsqsums = cs1.dsqsums[i] + cs2.dsqsums[i]

            variance = dsqsums / N - pow((dsums / N), 2)
            stds.append(math.sqrt(variance))
        return stds


class SummarizedSet:
    def __init__(self, points, center) -> None:
        self.num_points = len(points)
        self.num_dims = len(points[0])
        self.dsums = [0] * self.num_dims
        self.dsqsums = [0] * self.num_dims
        self.point_indices = set()
        
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

    def update(self, point):
        self.num_points += 1
        
        for i in range(self.num_dims):
            self.dsums[i] += point[i]
            self.dsqsums[i] += pow(point[i], 2)
            self.center[i] = self.dsums[i] / self.num_points

    def __str__(self) -> str:
        ret = []
        ret.append("------ Summarized Set Info ------")
        ret.append("Number of points: {}".format(self.num_points))
        ret.append("Number of dimensions: {}".format(self.num_dims))
        ret.append("Dimension sums: {}".format(self.dsums))
        ret.append("Dimension squared sums: {}".format(self.dsqsums))
        ret.append("Standard deviations: {}".format(self.get_stds()))
        ret.append("~~~~~~~~~~~~~~~~~~~~~")
        return "\n".join(ret)

    def merge(self, other):
        self.num_points += other.num_points
        self.point_indices |= other.point_indices
        
        for i in range(self.num_dims):
            self.dsums[i] += other.dsums[i]
            self.dsqsums[i] += other.dsqsums[i]
            self.center[i] = self.dsums[i] / self.num_points


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
        self.retained_set = []
        self.out_dict = {}
    
    def load_points(self, file_path):
        with open(file_path) as f:
            lines = f.readlines()
            for idx in range(len(lines)):
                line = lines[idx]
                pidx, pt = line.split(",", maxsplit=1)
                pt = list(map(float, pt.split(",")))
                lines[idx] = (pidx, pt)
        return lines

    def init_summarized_sets(self, pwi, labels, centers, summarized_sets):
        N = len(pwi)
        ss_points = defaultdict(list)
        seen_labels = set()

        for i in range(N):
            pidx = pwi[i][0]
            point = pwi[i][1]
            label = labels[pidx]
            seen_labels.add(label)
            ss_points[label].append(point)
        
        for label in seen_labels:
            summarized_sets.append(SummarizedSet(ss_points[label], centers[label]))

    def init_compressed_sets(self, pwi, labels, centers):
        N = len(pwi)
        ss_points = defaultdict(list)
        point_indices_per_set = defaultdict(set)
        seen_labels = set()

        for i in range(N):
            pidx = pwi[i][0]
            point = pwi[i][1]
            label = labels[pidx]
            seen_labels.add(label)
            ss_points[label].append(point)
            point_indices_per_set[label].add(pidx)
        
        for label in seen_labels:
            self.compressed_sets.append(SummarizedSet(ss_points[label], centers[label]))
            self.compressed_sets[-1].point_indices = point_indices_per_set[label]

    def merge_css(self):
        N = len(self.compressed_sets)
        print("Merging compressed sets of size: {}".format(N))
        kvs = {idx : x for idx, x in enumerate(self.compressed_sets)}

        can_merge = True
        num_merged = 0

        while can_merge and len(kvs) > 1:
            can_merge = False

            for cs1k, cs2k in combinations(kvs, 2):
                cs1 = kvs[cs1k]
                cs2 = kvs[cs2k]
                mhdist = Utils.mahalanobis_distance(cs1, cs2, cs2.get_stds())
                num_dims = cs2.num_dims

                
                # if self.cs_near(cs1, cs2, 2):
                if mhdist < 3 * math.sqrt(num_dims):
                    cs1.merge(cs2)
                    kvs.pop(cs2k)
                    kvs[cs1k] = cs1
                    can_merge = True
                    num_merged += 1
                    break
        css = []
        for cs in kvs.values():
            css.append(cs)
        self.compressed_sets = css
        print("Merged these many CS: {}".format(num_merged))

    def merge_into_ds(self):
        print("Number of retained set points at end: {}".format(len(self.retained_set)))
        for pidx, point in self.retained_set:
            label = self.assign_to_ss(point, self.discard_sets, 4)

            if label is not None:
                self.discard_sets[label].update(point)
                self.out_dict[pidx] = label
            else:
                self.out_dict[pidx] = -1
        
        for cs in self.compressed_sets:
            center = cs.center
            label = self.assign_to_ss(center, self.discard_sets, 10 ** 18)

            for pidx in cs.point_indices:
                self.out_dict[pidx] = label
            self.discard_sets[label].merge(cs)

    def update_SSs(self, assignments, ss):
        # assignments: idx, point, label
        for pidx, point, label in assignments:
            ss[label].update(point)
            ss[label].point_indices.add(pidx)

    def assign_to_ss(self, point, summarized_sets, alpha):
        choice = None
        smallest_dist = Utils.INF

        for i in range(len(summarized_sets)):
            ss = summarized_sets[i]
            dist = Utils.mahalanobis_distance(point, ss.center, ss.get_stds())

            if dist < smallest_dist and dist < alpha * math.sqrt(ss.num_dims):
                smallest_dist = dist
                choice = i
        return choice

    def init_DSs(self, ds_points, ds_centers):
        for label in ds_points:
            points = ds_points[label]
            center = ds_centers[label]
            self.discard_sets.append(SummarizedSet(points, center))

    def assign_dsrsout(self, points, alpha=3):
        for point in points:
            label = self.assign_to_ss(point[1], self.discard_sets, alpha)

            if label:
                self.discard_sets[label].update(point[1])
                self.out_dict[point[0]] = label
            else:
                self.retained_set.append(point)

    def init_sets(self, points):
        N = len(points)

        sample = math.ceil(0.2 * N)
        points_sample = points[:sample]
        points_rest = points[sample:]

        clst1 = HCluster(self.num_clusters * 3, 5)
        clst1.fit(points_sample)

        ctr = Counter(clst1.labels.values())
        best = ctr.most_common(self.num_clusters)

        best_labels = set()
        for k, _ in best:
            best_labels.add(k)

        ds_points = defaultdict(list)
        rem_points = []
        outlier_labels = set()
        for k, v in ctr.items():
            if v == 1:
                outlier_labels.add(k)

        label_map = {}
        ds_cluster_centers = {}

        for idx, (label, _) in enumerate(best):
            label_map[label] = idx
            ds_cluster_centers[idx] = clst1.cluster_centers[label]
        
        for i in range(sample):
            pidx, pt = points_sample[i]
            label = clst1.labels[pidx]

            if label in outlier_labels:
                self.retained_set.append(points_sample[i])
            elif label in best_labels:
                ds_points[label_map[label]].append(pt)
                self.out_dict[pidx] = label_map[label]
            else:
                rem_points.append(points_sample[i])
        
        self.init_DSs(ds_points, ds_cluster_centers)

        rem_points.extend(points_rest)
        self.assign_dsrsout(rem_points)

        for ds in self.discard_sets:
            print("{}".format(ds.num_points))

    def run(self):
        files = list(sorted(os.listdir(self.input_path)))
        print("Files: {}".format(files))

        for idx, file_name in enumerate(files):
            file_path = os.path.join(self.input_path, file_name)
            print("Processing index: {}/{}".format(idx + 1, len(files)))
            points = self.load_points(file_path)
            
            # For the first round, we want to do some extra stuff
            if idx == 0:
                self.init_sets(points)
            else:
                self.assign_dsrsout(points)

                # Do last file stuff
                if idx == len(files) - 1:
                    self.merge_into_ds()

            print("Number of assigned points in out dict: {}".format(len(self.out_dict)))
        
        with open(self.cluster_out, "w+") as f:
            json.dump(self.out_dict, f)
        with open(self.intermediate_out, "w+") as f:
            f.write(",".join(self.intermediate_header))
        
        print("Points per DS:")
        for ds in self.discard_sets:
            print(ds.num_points)


if __name__ == "__main__":
    st = time.time()

    runner = Runner()
    runner.run()

    print("Duration: {}".format(time.time() - st))
