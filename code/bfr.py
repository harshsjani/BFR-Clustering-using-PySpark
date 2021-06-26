import csv
import json
import math
import os
import random
import sys
import time

from itertools import combinations
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
        used_points.add(pwi[init_idx][0])

        for i in range(1, num_clusters):
            maxdist = 0
            new_centroid = None
            used_idx = None

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
                    used_idx = idx

            centroids[i] = new_centroid
            used_points.add(used_idx)
        return centroids

    def get_centroids_random(self, pwi, num_clusters):
        centroids = []
        seen = set()
        centers = 0

        while centers < num_clusters:
            rnd = random.Random()
            pt = rnd.choice(pwi)
            pidx = pt[0]
            pt = pt[1]

            if not pidx in seen:
                seen.add(pidx)
                centers += 1
                centroids.append(pt)
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
        
        ctr = Counter(labels.values())
        mvalue = min(ctr.values())
        print("Clustering mValues: {}".format(ctr))

        return centers, labels, mvalue

    def fit(self, pwi, randomized=False):
        best_value = None

        for _ in range(self.num_seeds):
            st = time.time()
            
            if randomized:
                centers = self.get_centroids_random(pwi, self.num_clusters)
            else:
                centers = self.get_centroids(pwi, self.num_clusters)
            print("Time to get centroids: {}".format(time.time() - st))

            centers, labels, mvalue = self.run(pwi, centers)

            if best_value is None or mvalue > best_value:
                best_labels = labels
                best_centers = centers
                best_value = mvalue

        self.cluster_centers = best_centers
        self.labels = best_labels
        self.best_value = best_value


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

    def cs_near(self, cs1, cs2, alpha):
        num_dims = cs1.num_dims
        stds1 = cs1.get_stds()
        stds2 = cs2.get_stds()
        union_stds = Utils.union_stds(cs1, cs2)

        for i in range(num_dims):
            std1 = stds1[i]
            std2 = stds2[i]
            stdsum = std1 + std2
            
            if not (union_stds[i] < alpha * (stdsum)):
                return False
        return True

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

    def cluster_rs(self):
        clustering = HCluster(self.num_clusters * 3, num_seeds=1, num_iterations=5)
        print("Clustering RS now.")
        print("Retained set: {}".format(self.retained_sets))
        clustering.fit(self.retained_sets)

        ctr = Counter(clustering.labels.values())

        init_cs_points = []
        cs_centers = {}
        outlier_points = []

        for idx, point in self.retained_sets:
            label = clustering.labels[idx]
            if ctr[label] == 1:
                outlier_points.append((idx, point))
            else:
                init_cs_points.append((idx, point))
                cs_centers[label] = clustering.cluster_centers[label]

        # Add more CS
        self.init_compressed_sets(init_cs_points, clustering.labels, cs_centers)

        # Reuse RS
        self.retained_sets = outlier_points

    def merge_into_ds(self):
        print("Number of retained set points at end: {}".format(len(self.retained_sets)))
        for pidx, point in self.retained_sets:
            label = Runner.assign_to_ss(point, self.discard_sets, 4)

            if label is not None:
                self.discard_sets[label].update(point)
            else:
                self.out_dict[pidx] = -1
        
        for cs in self.compressed_sets:
            center = cs.center
            label = Runner.assign_to_ss(center, self.discard_sets, 10 ** 18)

            for pidx in cs.point_indices:
                self.out_dict[pidx] = label
            self.discard_sets[label].merge(cs)

    def init_sets5(self, pointsRDD):
        pwi = pointsRDD.collect()
        num_points = len(pwi)
        print("Total number of points: {}".format(num_points))
        
        clustering = HCluster(self.num_clusters, num_seeds=10, num_iterations=1)
        clustering.fit(pwi, True)

        ctr = Counter(clustering.labels.values())
        print("New cluster counts: {}".format(ctr))

        self.init_summarized_sets(pwi, clustering.labels, clustering.cluster_centers, self.discard_sets)

        print ("DS point counts:")
        for ds in self.discard_sets:
            print(ds.num_points)

        # Write initial assignment
        for idx, _ in pwi:
            label = clustering.labels[idx]
            self.out_dict[idx] = label

    def init_sets4(self, pointsRDD):
        pwi = pointsRDD.collect()
        num_points = len(pwi)
        print("Total number of points: {}".format(num_points))
        
        fraction = math.ceil(0.2 * num_points)
        processed_points = 0
        inlier_points = []
        outlier_points = []

        pwi_sample = pwi[processed_points: processed_points + fraction]

        clustering = HCluster(self.num_clusters * 5, num_seeds=1, num_iterations=2)
        clustering.fit(pwi_sample)

        ctr = Counter(clustering.labels.values())
        print("New cluster counts: {}".format(ctr))

        outlier_labels = set()

        for k, v in ctr.items():
            if v == 1:
                outlier_labels.add(k)

        for idx, point in pwi_sample:
            label = clustering.labels[idx]
            if label in outlier_labels:
                outlier_points.append((idx, point))
            else:
                inlier_points.append((idx, point))

        inlier_points.extend(pwi[processed_points:])
        new_clusters = HCluster(self.num_clusters, num_seeds=1, num_iterations=2)
        new_clusters.fit(inlier_points)

        print("Number of label keys: {}".format(len(new_clusters.labels)))
        self.init_summarized_sets(inlier_points, new_clusters.labels, new_clusters.cluster_centers, self.discard_sets)

        # Finally, the RS
        self.retained_sets.extend(outlier_points)

        print ("DS point counts:")
        for ds in self.discard_sets:
            print(ds.num_points)

        # Write initial assignment
        for idx, _ in inlier_points:
            label = new_clusters.labels[idx]
            self.out_dict[idx] = label

    def init_sets3(self, pointsRDD):
        pwi = pointsRDD.collect()
        num_points = len(pwi)
        print("Total number of points: {}".format(num_points))
        
        clustering = HCluster(self.num_clusters * 5, num_seeds=1, num_iterations=2)
        clustering.fit(pwi)

        ctr = Counter(clustering.labels.values())
        print("New cluster counts: {}".format(ctr))

        outlier_labels = set()

        for k, v in ctr.items():
            if v == 1:
                outlier_labels.add(k)
        
        inlier_points = []
        outlier_points = []

        for idx, point in pwi:
            label = clustering.labels[idx]
            if label in outlier_labels:
                outlier_points.append((idx, point))
            else:
                inlier_points.append((idx, point))

        new_clusters = HCluster(self.num_clusters, num_seeds=1, num_iterations=2)
        new_clusters.fit(inlier_points)

        print("Number of label keys: {}".format(len(new_clusters.labels)))
        self.init_summarized_sets(inlier_points, new_clusters.labels, new_clusters.cluster_centers, self.discard_sets)

        # Now init the CSs
        # self.init_compressed_sets(init_cs_points, clustering.labels, cs_centers)

        # Finally, the RS
        self.retained_sets.extend(outlier_points)

        print ("DS point counts:")
        for ds in self.discard_sets:
            print(ds.num_points)

        # Write initial assignment
        for idx, _ in inlier_points:
            label = new_clusters.labels[idx]
            self.out_dict[idx] = label

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
        print("Cluster counts: {}".format(ctr))

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

        print("Number of inlier points + rest: {}".format(len(inlier_points)))
        clustering = HCluster(self.num_clusters * 2, num_seeds=1, num_iterations=5)
        clustering.fit(inlier_points)

        ctr = Counter(clustering.labels.values())

        print("New cluster counts: {}".format(ctr))
        most_common_labels = set(map(lambda x: x[0], ctr.most_common(self.num_clusters)))
        print("Most common labels: " + str(most_common_labels))
        init_ss_points = []
        init_cs_points = []
        ds_centers = {x : clustering.cluster_centers[x] for x in most_common_labels}
        cs_centers = {}

        for idx, point in inlier_points:
            label = clustering.labels[idx]
            if label in most_common_labels:
                init_ss_points.append((idx, point))
            elif ctr[label] == 1:
                outlier_points.append((idx, point))
            else:
                init_cs_points.append((idx, point))
                cs_centers[label] = clustering.cluster_centers[label]

        print("Number of label keys: {}".format(len(clustering.labels)))
        self.init_summarized_sets(init_ss_points, clustering.labels, ds_centers, self.discard_sets)

        # Now init the CSs
        self.init_summarized_sets(init_cs_points, clustering.labels, cs_centers, self.compressed_sets)

        # Finally, the RS
        self.retained_sets.extend(outlier_points)

        print ("DS point counts:")
        for ds in self.discard_sets:
            print(ds.num_points)

        # Write initial assignment
        for idx, _ in inlier_points:
            label = clustering.labels[idx]
            self.out_dict[idx] = label

    def update_SSs(self, assignments, ss):
        # assignments: idx, point, label
        for pidx, point, label in assignments:
            ss[label].update(point)
            ss[label].point_indices.add(pidx)

    @staticmethod
    def assign_to_ss(point, summarized_sets, alpha):
        choice = None
        smallest_dist = Utils.INF

        for i in range(len(summarized_sets)):
            ss = summarized_sets[i]
            dist = Utils.mahalanobis_distance(point, ss.center, ss.get_stds())

            if dist < smallest_dist and dist < alpha * math.sqrt(ss.num_dims):
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
                self.init_sets5(pointsRDD)
            else:
                dss = self.discard_sets
                css = self.compressed_sets

                assigned = pointsRDD.map(lambda idx_point: (
                    idx_point[0], idx_point[1], Runner.assign_to_ss(idx_point[1], dss, alpha=3)))
                
                # rem_for_cs_points = assigned.filter(lambda row: row[2] == None)
                
                ds_assigned = assigned.filter(lambda row: row[2] != None)
                self.out_dict.update(ds_assigned.map(lambda x: (x[0], x[2])).collectAsMap())
                
                # Update the discard sets now
                self.update_SSs(ds_assigned.collect(), self.discard_sets)

                # cs_assignment = rem_for_cs_points.map(lambda idx_point: (
                #     idx_point[0], idx_point[1], Runner.assign_to_ss(idx_point[1], css, alpha=3)))

                # cs_assigned = cs_assignment.filter(lambda row: row[2] != None)
                # self.update_SSs(cs_assigned.collect(), self.compressed_sets)
                
                rem_points = ds_assigned.filter(lambda row: row[2] == None).map(lambda x: (x[0], x[1])).collect()
                self.retained_sets.extend(rem_points)

                # Step 12
                # self.merge_css()

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
