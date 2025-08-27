import numpy as np
# from time import time

#######################
### RANDOM SAMPLING ###
#######################

def check_quadrats_overlap(point1, point2, quadrat_size_meters, pixel_size):
    """
    Check if two quadrats overlap based on their center points

    Parameters:
    -----------
    point1, point2 : tuple
        (y, x) coordinates of quadrat centers
    quadrat_size_meters : float
        Size of quadrats in meters
    pixel_size : float
        Size of each pixel in meters

    Returns:
    --------
    bool : True if quadrats overlap
    """
    y1, x1 = point1
    y2, x2 = point2

    quadrat_size_pixels = int(quadrat_size_meters / pixel_size)
    half_size = quadrat_size_pixels // 2

    # Check if quadrats overlap (center distance < quadrat size)
    distance = ((y1 - y2) ** 2 + (x1 - x2) ** 2) ** 0.5
    return distance < quadrat_size_pixels


def random_sampling_map(img, n_samples, quadrat_size_meters=1.0, pixel_size=0.01):
    """
    Perform random sampling with non-overlapping quadrats (simple version)

    Parameters:
    -----------
    img : numpy.ndarray
        Input image to sample from
    n_samples : int
        Number of samples to generate
    quadrat_size_meters : float, optional
        Size of sampling window in meters (default: 1.0)
    pixel_size : float, optional
        Size of each pixel in meters (default: 0.01)

    Returns:
    --------
    sample_points : numpy.ndarray
        Array of (y, x) coordinates for the center of each sample
    samples : list
        List of sample arrays, each of size window_size × window_size
    """
    # Calculate window size in pixels
    window_pixels = int(quadrat_size_meters / pixel_size)
    padding = window_pixels // 2

    # Get valid coordinate ranges
    y_size = img.shape[0] - 2 * padding
    x_size = img.shape[1] - 2 * padding

    if y_size <= 0 or x_size <= 0:
        print("Warning: Image too small for quadrat size")
        return np.array([]), []

    sample_points = []

    for _ in range(n_samples):
        valid_point_found = False
        attempts = 0

        while not valid_point_found and attempts < 1000:
            attempts += 1

            # Generate random position
            sample_y = np.random.randint(padding, img.shape[0] - padding)
            sample_x = np.random.randint(padding, img.shape[1] - padding)
            candidate_point = (sample_y, sample_x)

            # Check overlap with existing points
            overlaps = False
            for existing_point in sample_points:
                if check_quadrats_overlap(candidate_point, existing_point,
                                          quadrat_size_meters, pixel_size):
                    overlaps = True
                    break

            if not overlaps:
                valid_point_found = True
                sample_points.append(candidate_point)

        if not valid_point_found:
            print(
                f"Warning: Could not generate {n_samples} non-overlapping quadrats. Only {len(sample_points)} were created.")
            break

    # Convert to numpy array
    sample_points = np.array(sample_points)

    # Extract centered samples
    samples = []
    for y, x in sample_points:
        sample = img[y - padding:y + padding, x - padding:x + padding]
        samples.append(sample)

    return sample_points, samples

def generate_samples_MC_random(img, n_samples, n_MC, quadrat_size):
    samples = []
    for _ in range(n_MC):
        _, s = random_sampling_map(img, n_samples, quadrat_size)
        samples.append(s)
    samples = np.array(samples)
    return samples


##############################
### FREE TRANSECT SAMPLING ###
##############################

def sample_single_transect(img, n_quadrat_per_transect, distance_between_quadrats,
                           quadrat_size_meters=1.0, pixel_size=0.01):
    img_height, img_width = img.shape

    # Calculate quadrat size in pixels
    quadrat_size_pixels = int(quadrat_size_meters / pixel_size)
    padding = quadrat_size_pixels // 2

    # Calculate distance between quadrats in pixels
    quadrat_distance_pixels = int(distance_between_quadrats / pixel_size)

    # Step 1: Sample transect angle (0, pi)
    angle = np.random.uniform(0, np.pi)
    # print(f"Sampled angle: {np.rad2deg(angle):.1f} degrees")

    # Step 2: Calculate valid padding for first point (center quadrat)
    # We need to ensure the transect fits within the image
    y_min, y_max = padding + 1, img_height - padding - 1
    x_min, x_max = padding + 1, img_width - padding - 1

    # Calculate the total length of the transect in pixels
    transect_length_pixels = (n_quadrat_per_transect - 1) * quadrat_distance_pixels + quadrat_size_pixels

    # Find how much room we need to leave to ensure the transect fits
    max_y_extension = abs(transect_length_pixels * np.sin(angle))
    max_x_extension = abs(transect_length_pixels * np.cos(angle))

    # Adjust valid area for sampling the first point
    valid_y_min = y_min
    valid_y_max = min(y_max, img_height - padding - 1 - max_y_extension)
    valid_x_min = max(x_min, padding + 1 + max_x_extension) if np.cos(angle) < 0 else x_min
    valid_x_max = min(x_max, img_width - padding - 1 - max_x_extension) if np.cos(angle) > 0 else x_max

    # Step 3: Sample transect first point
    start_y = np.random.randint(valid_y_min, valid_y_max + 1)
    start_x = np.random.randint(valid_x_min, valid_x_max + 1)
    # print(f"Sampled start point: ({start_y}, {start_x})")

    # Step 4: Compute all quadrat centers along the transect
    quadrat_points = []
    for i in range(n_quadrat_per_transect):
        # Calculate position along transect
        offset = i * quadrat_distance_pixels
        y = int(start_y + offset * np.sin(angle))
        x = int(start_x + offset * np.cos(angle))
        quadrat_points.append((y, x))

    return quadrat_points

def check_transects_overlap(quadrat_points1, quadrat_points2, quadrat_size_meters, pixel_size):
    quadrat_size_pixels = int(quadrat_size_meters / pixel_size)

    # Find bounding rectangle for first transect
    y_coords1 = [y for y, x in quadrat_points1]
    x_coords1 = [x for y, x in quadrat_points1]

    min_y1 = min(y_coords1) - quadrat_size_pixels // 2
    max_y1 = max(y_coords1) + quadrat_size_pixels // 2
    min_x1 = min(x_coords1) - quadrat_size_pixels // 2
    max_x1 = max(x_coords1) + quadrat_size_pixels // 2

    # Find bounding rectangle for second transect
    y_coords2 = [y for y, x in quadrat_points2]
    x_coords2 = [x for y, x in quadrat_points2]

    min_y2 = min(y_coords2) - quadrat_size_pixels // 2
    max_y2 = max(y_coords2) + quadrat_size_pixels // 2
    min_x2 = min(x_coords2) - quadrat_size_pixels // 2
    max_x2 = max(x_coords2) + quadrat_size_pixels // 2

    # Check for rectangle overlap
    # Two rectangles overlap if they overlap on both x and y axes
    x_overlap = (min_x1 <= max_x2) and (max_x1 >= min_x2)
    y_overlap = (min_y1 <= max_y2) and (max_y1 >= min_y2)

    return x_overlap and y_overlap

def free_transect_sampling_map(img, n_transects,
                               n_quadrat_per_transect, distance_between_quadrats,
                               quadrat_size_meters=1.0, pixel_size=0.01):
    # Calculate quadrat size in pixels
    quadrat_size_pixels = int(quadrat_size_meters / pixel_size)
    padding = quadrat_size_pixels // 2

    all_transects = []
    for _ in range(n_transects):
        valid_transect_found = False
        attempts = 0

        while not valid_transect_found and attempts < 1000:
            attempts += 1

            transect_points = sample_single_transect(img, n_quadrat_per_transect, distance_between_quadrats)

            # Check overlap with previous transects
            overlaps = False
            for prev_transect_points in all_transects:
                if check_transects_overlap(transect_points, prev_transect_points, quadrat_size_meters, pixel_size):
                    overlaps = True
                    break

            if not overlaps:
                valid_transect_found = True
                all_transects.append(transect_points)

        if not valid_transect_found:
            print(
                f"Warning: Could not generate {n_transects} non-overlapping transects. Only {len(all_transects)} were created.")
            break

    sample_points = np.vstack(all_transects)

    # Extract centered samples
    samples = []
    for y, x in sample_points:
        sample = img[y - padding:y + padding, x - padding:x + padding]
        samples.append(sample)

    return sample_points, samples

def generate_samples_MC_free_transects(img, n_transects, n_quadrat_per_transect, distance_between_quadrats,
                                       n_MC, quadrat_size):
    samples = []
    while len(samples) < n_MC:
        _, s = free_transect_sampling_map(img, n_transects, n_quadrat_per_transect, distance_between_quadrats,
                                          quadrat_size)

        if len(s) >= n_transects:
            samples.append(s)

    samples = np.array(samples)
    return samples


##################################
### PARALLEL TRANSECT SAMPLING ###
##################################

def parallel_transect_sampling_map(img, n_transects,
                                   n_quadrat_per_transect, distance_between_quadrats, distance_between_transects,
                                   quadrat_size_meters=1.0, pixel_size=0.01):
    """
    Perform parallel transect sampling on an image

    Parameters:
    -----------
    img : numpy.ndarray
        Input image to sample from
    n_transects : int
        Number of parallel transects to generate
    n_quadrat_per_transect : int
        Number of quadrats per transect
    distance_between_quadrats : float
        Distance between quadrats along a transect in meters
    distance_between_transects : float
        Distance between parallel transects in meters
    quadrat_size_meters : float, optional
        Size of sampling quadrat in meters (default: 1.0)
    pixel_size : float, optional
        Size of each pixel in meters (default: 0.01)

    Returns:
    --------
    sample_points : numpy.ndarray
        Array of (y, x) coordinates for the center of each sample
    samples : list
        List of sample arrays, each of size quadrat_size × quadrat_size
    """

    img_height, img_width = img.shape

    # Calculate sizes in pixels
    quadrat_size_pixels = int(quadrat_size_meters / pixel_size)
    padding = quadrat_size_pixels // 2
    quadrat_distance_pixels = int(distance_between_quadrats / pixel_size)
    transect_distance_pixels = int(distance_between_transects / pixel_size)

    valid_transects_found = False
    attempts = 0

    while not valid_transects_found and attempts < 1000:
        attempts += 1

        angle = np.random.uniform(0, np.pi)

        # Calculate directions
        transect_direction_y = np.sin(angle)
        transect_direction_x = np.cos(angle)
        perp_direction_y = np.sin(angle + np.pi / 2)
        perp_direction_x = np.cos(angle + np.pi / 2)

        # Generate all quadrat centers (relative to origin)
        relative_quadrat_centers = []
        for t in range(n_transects):
            # Calculate the start point for this transect
            transect_start_y = t * transect_distance_pixels * perp_direction_y
            transect_start_x = t * transect_distance_pixels * perp_direction_x

            for q in range(n_quadrat_per_transect):
                # Calculate position along transect
                quadrat_y = transect_start_y + q * quadrat_distance_pixels * transect_direction_y
                quadrat_x = transect_start_x + q * quadrat_distance_pixels * transect_direction_x
                relative_quadrat_centers.append((quadrat_y, quadrat_x))

        # Find the bounding rectangle
        min_rel_y = min(y for y, x in relative_quadrat_centers) - padding - 1
        max_rel_y = max(y for y, x in relative_quadrat_centers) + padding + 1
        min_rel_x = min(x for y, x in relative_quadrat_centers) - padding - 1
        max_rel_x = max(x for y, x in relative_quadrat_centers) + padding + 1

        # Calculate valid regions for placing the first quadrat
        valid_y_min = int(0 - min_rel_y)
        valid_y_max = int(img_height - max_rel_y)
        valid_x_min = int(0 - min_rel_x)
        valid_x_max = int(img_width - max_rel_x)

        if valid_y_min <= valid_y_max and valid_x_min <= valid_x_max:
            valid_transects_found = True
            origin_y = np.random.randint(valid_y_min, valid_y_max + 1)
            origin_x = np.random.randint(valid_x_min, valid_x_max + 1)

    if not valid_transects_found:
        print("Cannot fit the parallel transects within the image. Try reducing the number of transects, "
              "the number of quadrats per transect, or the distances between them.")
        return [], []

    # Calculate actual quadrat centers
    all_quadrat_points = []
    for rel_y, rel_x in relative_quadrat_centers:
        y = int(origin_y + rel_y)
        x = int(origin_x + rel_x)
        all_quadrat_points.append((y, x))

    # Convert to numpy array
    sample_points = np.array(all_quadrat_points)

    # Extract centered samples
    samples = []
    for y, x in sample_points:
        sample = img[y - padding:y + padding, x - padding:x + padding]
        samples.append(sample)

    return sample_points, samples

def generate_samples_MC_parallel_transects(img, n_transects, n_quadrat_per_transect,
                                           distance_between_quadrats, distance_between_transects,
                                           n_MC, quadrat_size):
    samples = []
    while len(samples) < n_MC:
        _, s = parallel_transect_sampling_map(img, n_transects,
                                              n_quadrat_per_transect, distance_between_quadrats,
                                              distance_between_transects, quadrat_size)

        if len(s) >= n_transects:
            samples.append(s)

    samples = np.array(samples)
    return samples


##############################################
### NON DIRECTIONAL (ND) TRANSECT SAMPLING ###
##############################################

def sample_single_ND_transect(img, n_quadrat_per_transect, distance_between_quadrats,
                              quadrat_size_meters=1.0, pixel_size=0.01):
    """
    Sample a single non-directional transect where quadrats move in random directions

    Parameters:
    -----------
    img : numpy.ndarray
        Input image to sample from
    n_quadrat_per_transect : int
        Number of quadrats per transect
    distance_between_quadrats : float
        Approximate distance between quadrats in meters
    quadrat_size_meters : float, optional
        Size of sampling quadrat in meters (default: 1.0)
    pixel_size : float, optional
        Size of each pixel in meters (default: 0.01)
    max_attempts : int, optional
        Maximum number of attempts to place a single quadrat (default: 100)

    Returns:
    --------
    quadrat_points : list
        List of (y, x) coordinates for the center of each quadrat
    """
    img_height, img_width = img.shape

    # Calculate sizes in pixels
    quadrat_size_pixels = int(quadrat_size_meters / pixel_size)
    padding = quadrat_size_pixels // 2
    quadrat_distance_pixels = int(distance_between_quadrats / pixel_size)

    valid_y_min = padding + 1
    valid_y_max = img_height - padding - 1
    valid_x_min = padding + 1
    valid_x_max = img_width - padding - 1

    # Sample the first quadrat center
    first_y = np.random.randint(valid_y_min, valid_y_max + 1)
    first_x = np.random.randint(valid_x_min, valid_x_max + 1)

    quadrat_points = [(first_y, first_x)]

    # Sample the rest of the quadrats in random directions
    for i in range(1, n_quadrat_per_transect):
        previous_y, previous_x = quadrat_points[-1]

        # Try different directions until we find a valid one
        attempts = 0
        quadrat_placed = False

        while not quadrat_placed and attempts < 1000:
            attempts += 1

            # Sample random direction
            angle = np.random.uniform(0, 2 * np.pi)

            # Calculate new position
            new_y = int(previous_y + quadrat_distance_pixels * np.sin(angle))
            new_x = int(previous_x + quadrat_distance_pixels * np.cos(angle))

            # Check if it's within image bounds with padding
            if (new_y - padding >= 0 and new_y + padding < img_height and
                    new_x - padding >= 0 and new_x + padding < img_width):

                # Check for overlap with existing quadrats
                overlaps = False
                for qy, qx in quadrat_points:
                    # Calculate distance between centers
                    center_distance = np.sqrt((qy - new_y) ** 2 + (qx - new_x) ** 2)
                    # If centers are closer than quadrat size, there's overlap
                    if center_distance < quadrat_size_pixels:
                        overlaps = True
                        break

                if not overlaps:
                    quadrat_points.append((new_y, new_x))
                    quadrat_placed = True

        # If we couldn't place this quadrat after many attempts, raise an exception
        if not quadrat_placed:
            raise ValueError(f"Could not place quadrat {i + 1} after {1000} attempts")

    return quadrat_points


def ND_transect_sampling_map(img, n_transects, n_quadrat_per_transect, distance_between_quadrats,
                             quadrat_size_meters=1.0, pixel_size=0.01):
    """
    Perform non-directional transect sampling where each transect consists of quadrats
    placed in random directions from each other, avoiding overlaps between transects.

    Parameters:
    -----------
    img : numpy.ndarray
        Input image to sample from
    n_transects : int
        Number of non-directional transects to generate
    n_quadrat_per_transect : int
        Number of quadrats per transect
    distance_between_quadrats : float
        Approximate distance between quadrats in meters
    quadrat_size_meters : float, optional
        Size of sampling quadrat in meters (default: 1.0)
    pixel_size : float, optional
        Size of each pixel in meters (default: 0.01)

    Returns:
    --------
    sample_points : numpy.ndarray
        Array of (y, x) coordinates for the center of each sample
    samples : list
        List of sample arrays, each of size quadrat_size × quadrat_size
    """

    # Calculate sizes in pixels
    quadrat_size_pixels = int(quadrat_size_meters / pixel_size)
    padding = quadrat_size_pixels // 2

    all_transects = []

    for _ in range(n_transects):
        valid_transect_found = False
        attempts = 0

        while not valid_transect_found and attempts < 100:
            attempts += 1

            try:
                # Try to generate a single non-directional transect
                success = False
                while not success:
                    try:
                        transect_points = sample_single_ND_transect(
                            img, n_quadrat_per_transect, distance_between_quadrats,
                            quadrat_size_meters, pixel_size)
                        success = True
                    except ValueError:
                        continue

                # Check overlap with previous transects
                overlaps = False
                for prev_transect_points in all_transects:
                    if check_transects_overlap(transect_points, prev_transect_points, quadrat_size_meters, pixel_size):
                        overlaps = True
                        break

                if not overlaps:
                    valid_transect_found = True
                    all_transects.append(transect_points)
            except ValueError:
                # If we couldn't place a quadrat, try again with a different starting point
                continue

        if not valid_transect_found:
            print(
                f"Warning: Could not generate {n_transects} non-overlapping transects.")
            return [], []

    # Combine all transect points
    sample_points = np.vstack(all_transects) #if all_transects else np.array([])

    # Extract centered samples
    samples = []
    for y, x in sample_points:
        sample = img[y - padding:y + padding, x - padding:x + padding]
        samples.append(sample)

    return sample_points, samples


def generate_samples_MC_ND_transects(img, n_transects, n_quadrat_per_transect,
                                     distance_between_quadrats, n_MC, quadrat_size):
    """
    Generate multiple Monte Carlo samples using non-directional transect sampling

    Parameters:
    -----------
    img : numpy.ndarray
        Input image to sample from
    n_transects : int
        Number of non-directional transects to generate
    n_quadrat_per_transect : int
        Number of quadrats per transect
    distance_between_quadrats : float
        Approximate distance between quadrats in meters
    n_MC : int
        Number of Monte Carlo samples to generate
    quadrat_size : float
        Size of sampling quadrat in meters

    Returns:
    --------
    samples : numpy.ndarray
        Array of samples with shape (n_MC, n_transects*n_quadrat_per_transect, quadrat_size, quadrat_size)
    """
    samples = []
    while len(samples) < n_MC:
        _, s = ND_transect_sampling_map(img, n_transects, n_quadrat_per_transect,
                                            distance_between_quadrats, quadrat_size)
        if len(s) >= n_transects:
            samples.append(s)

    samples = np.array(samples)
    return samples


######################
### MODIFY SAMPLES ###
######################

def crop_quadrat_half(quadrat):
    """Extract the center portion of the quadrat (1/4 area)"""
    h, w = quadrat.shape
    h_start, h_end = h // 4, 3 * h // 4
    w_start, w_end = w // 4, 3 * w // 4
    return quadrat[h_start:h_end, w_start:w_end]

def get_half_quadrat_samples(samples_MC):
    samples_half = []
    for samples in samples_MC:
        samples_half.append(np.array([crop_quadrat_half(quadrat) for quadrat in samples]))
    samples_half = np.array(samples_half)
    return samples_half

def subsample_quadrats(samples_MC, n_quadrats):
    subsamples = samples_MC[:, :n_quadrats, :, :]
    return subsamples


###################################
### SIMULATE QUADRAT ANNOTATION ###
###################################

def sample_quadrats(quadrats, n_points):
    points = []
    for q in quadrats:
        x_coords = np.arange(q.shape[0])
        y_coords = np.arange(q.shape[1])
        coords = np.array(np.meshgrid(x_coords, y_coords)).T.reshape(-1, 2)

        indices = np.random.choice(len(coords), n_points, replace=False)
        pts = coords[indices]

        for p in pts:
            pt = q[p[0], p[1]]
            points.append(pt)

    points = np.array(points)
    return points

def sample_quadrats_streamlit(quadrats, n_points):
    points, points_locations = [], []
    for q in quadrats:
        points_locations.append([])
        x_coords = np.arange(q.shape[0])
        y_coords = np.arange(q.shape[1])
        coords = np.array(np.meshgrid(x_coords, y_coords)).T.reshape(-1, 2)

        indices = np.random.choice(len(coords), n_points, replace=False)
        pts = coords[indices]

        for p in pts:
            pt = q[p[0], p[1]]
            points.append(pt)
            points_locations[-1].append([p[0], p[1]])
        points_locations[-1] = np.array(points_locations[-1])
    points = np.array(points)

    return points, points_locations

def sample_quadrats_MC(MC_samples, n_points):
    samples_pts = []
    for samples in MC_samples:
        samples_pts.append(sample_quadrats(samples, n_points))
    samples_pts = np.array(samples_pts)

    return samples_pts


#####################################
### SIMULATE AUTOMATED ANNOTATION ###
#####################################

def simulate_predictions_by_flipping(true_labels, precision, recall, target_class, random_seed=None):
    """
    Simulate predictions by flipping labels based on precision and recall.

    For non-target class: P(predict target) = 1 - precision
    For target class: P(predict correct) = recall

    Parameters:
    -----------
    true_labels : array-like
        Ground truth labels
    precision : float
        Target precision (between 0 and 1)
    recall : float
        Target recall (between 0 and 1)
    target_class : int
        The class of interest
    random_seed : int
        Random seed for reproducibility

    Returns:
    --------
    predictions : array-like
        Simulated predictions
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    true_labels = np.array(true_labels)
    predictions = true_labels.copy()

    # Calculate FPR from precision and recall
    # precision = TP / (TP + FP)
    # Si on a N_pos positifs et N_neg négatifs:
    # TP = recall * N_pos
    # FP = FPR * N_neg
    # precision = (recall * N_pos) / (recall * N_pos + FPR * N_neg)

    target_mask = (true_labels == target_class)
    n_pos = np.sum(target_mask)
    n_neg = np.sum(~target_mask)

    if n_pos > 0 and n_neg > 0 and precision > 0:
        # Résoudre pour FPR
        fpr = (recall * n_pos * (1 - precision)) / (precision * n_neg)
        fpr = min(fpr, 1.0)  # Limiter à 1.0
    else:
        fpr = 0

    # Pour les instances de la classe cible : TPR = recall
    target_indices = np.where(target_mask)[0]
    for idx in target_indices:
        if np.random.random() < (1 - recall):  # Manquer avec prob (1-recall)
            # Choisir une classe non-cible aléatoire
            non_target_classes = np.unique(true_labels[~target_mask])
            if len(non_target_classes) > 0:
                predictions[idx] = np.random.choice(non_target_classes)
            else:
                predictions[idx] = target_class + 1

    # Pour les instances non-cible : FPR = FP/(FP+TN)
    non_target_indices = np.where(~target_mask)[0]
    for idx in non_target_indices:
        if np.random.random() < fpr:  # Faux positif avec prob FPR
            predictions[idx] = target_class

    return predictions

def simulate_predictions_MC(true_labels_MC, precision, recall, target_class, random_seed=None):
    preds = []
    for true_labels in true_labels_MC:
        preds.append(simulate_predictions_by_flipping(true_labels, precision, recall, target_class, random_seed))
    preds = np.array(preds)
    return preds


