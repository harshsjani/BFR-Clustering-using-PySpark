import random





def _kmeans_single_elkan(X, centers_init, max_iter=300):
    n_samples = X.shape[0]
    n_clusters = centers_init.shape[0]

    # Buffers to avoid new allocations at each iteration.
    centers = centers_init
    centers_new = np.zeros_like(centers)
    weight_in_clusters = np.zeros(n_clusters, dtype=X.dtype)
    labels = np.full(n_samples, -1, dtype=np.int32)
    labels_old = labels.copy()
    center_half_distances = euclidean_distances(centers) / 2
    distance_next_center = np.partition(np.asarray(center_half_distances),
                                        kth=1, axis=0)[1]

    [0] * n_samples
    upper_bounds = np.zeros(n_samples, dtype=X.dtype)
    lower_bounds = np.zeros((n_samples, n_clusters), dtype=X.dtype)
    center_shift = np.zeros(n_clusters, dtype=X.dtype)

    if sp.issparse(X):
        init_bounds = init_bounds_sparse
        elkan_iter = elkan_iter_chunked_sparse
        _inertia = _inertia_sparse
    else:
        init_bounds = init_bounds_dense
        elkan_iter = elkan_iter_chunked_dense
        _inertia = _inertia_dense

    init_bounds(X, centers, center_half_distances,
                labels, upper_bounds, lower_bounds)

    strict_convergence = False

    for i in range(max_iter):
        elkan_iter(X, sample_weight, centers, centers_new,
                   weight_in_clusters, center_half_distances,
                   distance_next_center, upper_bounds, lower_bounds,
                   labels, center_shift, n_threads)

        # compute new pairwise distances between centers and closest other
        # center of each center for next iterations
        center_half_distances = euclidean_distances(centers_new) / 2
        distance_next_center = np.partition(
            np.asarray(center_half_distances), kth=1, axis=0)[1]

        if verbose:
            inertia = _inertia(X, sample_weight, centers, labels)
            print(f"Iteration {i}, inertia {inertia}")

        centers, centers_new = centers_new, centers

        if np.array_equal(labels, labels_old):
            # First check the labels for strict convergence.
            if verbose:
                print(f"Converged at iteration {i}: strict convergence.")
            strict_convergence = True
            break
        else:
            # No strict convergence, check for tol based convergence.
            center_shift_tot = (center_shift**2).sum()
            if center_shift_tot <= tol:
                if verbose:
                    print(f"Converged at iteration {i}: center shift "
                          f"{center_shift_tot} within tolerance {tol}.")
                break

        labels_old[:] = labels

    if not strict_convergence:
        # rerun E-step so that predicted labels match cluster centers
        elkan_iter(X, sample_weight, centers, centers, weight_in_clusters,
                   center_half_distances, distance_next_center,
                   upper_bounds, lower_bounds, labels, center_shift,
                   n_threads, update_centers=False)

    inertia = _inertia(X, sample_weight, centers, labels)

    return labels, inertia, centers, i + 1
