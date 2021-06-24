import json
from sklearn.metrics import normalized_mutual_info_score
import os


gt_paths = [
    "../resource/asnlib/publicdata/cluster1.json",
    "../resource/asnlib/publicdata/cluster2.json",
    "../resource/asnlib/publicdata/cluster3.json",
    "../resource/asnlib/publicdata/cluster4.json",
    "../resource/asnlib/publicdata/cluster5.json",
]

my_paths = [
    "test1_clust_out",
    "test2_clust_out",
    "test3_clust_out",
    "test4_clust_out",
    "test5_clust_out",
]

selected_idx = 0

ground_truth_labels_path = gt_paths[selected_idx]
my_label_path = my_paths[selected_idx]

with open(ground_truth_labels_path) as f:
    ground_truth_labels = json.load(f)
with open(my_label_path) as f:
    my_labels = json.load(f)

print("Number of ground truth assignments: {}".format(len(ground_truth_labels)))
print("Number of predicted assignments: {}".format(len(my_labels)))

gt_feature = [-1] * len(ground_truth_labels)
for k, v in ground_truth_labels.items():
    gt_feature[int(k)] = v

my_feature = [-1] * len(my_labels)
for k, v in my_labels.items():
    my_feature[int(k)] = v

normalized_mutual_info = normalized_mutual_info_score(gt_feature, my_feature)
print("The nmi is {}".format(normalized_mutual_info))
