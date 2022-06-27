import tensorflow as tf

def _fix_up_point_sets(point_set_a, point_set_b):
    """
    Extract only the x,y,z positions from both the prediction and the ground truth

    point_set_a: ground truth
    point_set_b: prediction
    """
    point_set_a = tf.convert_to_tensor(value=point_set_a)
    point_set_b = tf.convert_to_tensor(value=point_set_b)
    
    # keep only spatial information
    point_set_a = point_set_a[...,:3]
    point_set_b = point_set_b[...,:3]
    return point_set_a, point_set_b

def _select_by_marker(point_set_a, point_set_b):
    """
    Keep only points (both predicted and ground truth) whose marker is enabled

    point_set_a: ground truth
    point_set_b: prediction
    """
    point_set_a = tf.convert_to_tensor(value=point_set_a)
    point_set_b = tf.convert_to_tensor(value=point_set_b)

    enabled_points = point_set_a[...,-1]
    mask = tf.greater(enabled_points, 0)
    return point_set_a[mask], point_set_b[mask]

def chamfer_loss(point_set_a, point_set_b, name="chamfer"):
    """
    1:1 copy from https://www.tensorflow.org/graphics/api_docs/python/tfg/nn/loss/chamfer_distance/evaluate
    with included fix up of points.
    The Chamfer distance is calculated as the sum of the average minimum distance from point_set_a to 
    point_set_b and vice versa. The average minimum distance from one point set to another is calculated as 
    the average of the distances between the points in the first set and their closest point in the second set, 
    and is thus not symmetrical.
    Note that we want the points in set 'a' to be near all points in 'b' (i.e. avoid that all points in 'a' 
    converge to a single point of 'b' as it would happen with a symmetrical loss)

    point_set_a: ground truth
    point_set_b: prediction
    """
    with tf.name_scope(name):
        
        point_set_a, point_set_b = _fix_up_point_sets(point_set_a, point_set_b)
        
        # Create N x M matrix where the entry i,j corresponds to ai - bj (vector of dimension D).
        difference = (tf.expand_dims(point_set_a, axis=-2) - tf.expand_dims(point_set_b, axis=-3))
        # Calculate the square distances between each two points: |ai - bj|^2.
        square_distances = tf.einsum("...i,...i->...", difference, difference)

        minimum_square_distance_a_to_b = tf.reduce_min(input_tensor=square_distances, axis=-1)
        minimum_square_distance_b_to_a = tf.reduce_min(input_tensor=square_distances, axis=-2)

        return (
            tf.reduce_mean(input_tensor=minimum_square_distance_a_to_b, axis=-1) +
            tf.reduce_mean(input_tensor=minimum_square_distance_b_to_a, axis=-1))

def selective_chamfer_loss(point_set_a, point_set_b, name="selective_chamfer"):
    """
    Apply Chamfer loss only to enabled markers

    point_set_a: ground truth
    point_set_b: prediction
    """
    point_set_a, _ = _select_by_marker(point_set_a, point_set_b)
    return chamfer_loss(point_set_a, point_set_b, name)
    
def hausdorff_loss(point_set_a, point_set_b, name="hausdorff"):
    """
    1:1 copy from https://www.tensorflow.org/graphics/api_docs/python/tfg/nn/loss/hausdorff_distance/evaluate
    with included fix up of points.
    The Hausdorff distance from point_set_a to point_set_b is defined as the maximum of all distances from a 
    point in point_set_a to the closest point in point_set_b. It is an asymmetric metric.

    point_set_a: ground truth
    point_set_b: prediction
    """
    with tf.name_scope(name):
        point_set_a, point_set_b = _fix_up_point_sets(point_set_a, point_set_b)

        # Create N x M matrix where the entry i,j corresponds to ai - bj (vector of
        # dimension D).
        difference = (tf.expand_dims(point_set_a, axis=-2) - tf.expand_dims(point_set_b, axis=-3))
        # Calculate the square distances between each two points: |ai - bj|^2.
        square_distances = tf.einsum("...i,...i->...", difference, difference)

        minimum_square_distance_a_to_b = tf.reduce_min(input_tensor=square_distances, axis=-1)
        return tf.sqrt(tf.reduce_max(input_tensor=minimum_square_distance_a_to_b, axis=-1))

def selective_hausdorff_loss(point_set_a, point_set_b, name="selective_hausdorff"):
    """
    Apply Hausdorff loss only to enabled markers

    point_set_a: ground truth
    point_set_b: prediction
    """
    point_set_a, _ = _select_by_marker(point_set_a, point_set_b)
    return hausdorff_loss(point_set_a, point_set_b, name)

def perpoint_squared_distance(point_set_a, point_set_b, name="squared_dist"):
    """
    Per-point squared distance is the squared Euclidean distance between each ordered pair of points 
    from the input sets. The distance is computed between points with the same index:
    ppd(a[0], b[0])+ppd(a[1], b[1])+...

    point_set_a: ground truth
    point_set_b: prediction
    """
    with tf.name_scope(name):
        
        point_set_a, point_set_b = _fix_up_point_sets(point_set_a, point_set_b)

        dist = (point_set_a - point_set_b) ** 2 # (x_a - x_b)^2, (y_a - y_b)^2, (z_a - z_b)^2
        dist = tf.reduce_sum(dist, axis = -1) # sum over xyz axis, i.e. (x_a - x_b)^2 + (y_a - y_b)^2 + (z_a - z_b)^2
        dist = tf.reduce_mean(dist, axis = -1) # mean for all points in a frame (avoids incomparable losses when a different number of markers is enabled)
        return dist

def selective_perpoint_squared_distance(point_set_a, point_set_b, name="selective_squared_dist"):
    """
    Apply Per-point squared loss only to enabled markers

    point_set_a: ground truth
    point_set_b: prediction
    """
    point_set_a, point_set_b = _select_by_marker(point_set_a, point_set_b)
    return perpoint_squared_distance(point_set_a, point_set_b, name)

def selective_perpoint_distance(point_set_a, point_set_b, name="selective_dist"):
    """
    Apply Per-point Euclidean loss only to enabled markers

    point_set_a: ground truth
    point_set_b: prediction
    """
    point_set_a, point_set_b = _select_by_marker(point_set_a, point_set_b)
    return tf.sqrt(perpoint_squared_distance(point_set_a, point_set_b, name))

def perpoint_manhattan_distance(point_set_a, point_set_b, name="manhattan_dist"):
    """
    Per-point manhattan distance is the manhattan distance between each ordered pair of points 
    from the input sets. The distance is computed between points with the same index:
    ppd(a[0], b[0])+ppd(a[1], b[1])+...

    point_set_a: ground truth
    point_set_b: prediction
    """
    with tf.name_scope(name):
        point_set_a, point_set_b = _fix_up_point_sets(point_set_a, point_set_b)
        dist = tf.math.abs(point_set_a - point_set_b) # abs(x_a - x_b), (y_a - y_b), (z_a - z_b)
        dist = tf.reduce_sum(dist, axis = -1) # sum over xyz axis, i.e. (x_a - x_b)^2 + (y_a - y_b)^2 + (z_a - z_b)^2
        dist = tf.reduce_mean(dist, axis = -1) # mean for all points in a frame (avoids incomparable losses when a different number of markers is enabled)
        return dist

def selective_perpoint_manhattan_distance(point_set_a, point_set_b, name="selective_manhattan_dist"):
    """
    Apply Per-point manhattan loss only to enabled markers

    point_set_a: ground truth
    point_set_b: prediction
    """
    point_set_a, point_set_b = _select_by_marker(point_set_a, point_set_b)
    return perpoint_manhattan_distance(point_set_a, point_set_b, name)

def perpoint_cosine_distance(point_set_a, point_set_b, name="cosine_dist"):
    """
    Per-point cosine distance is the cosine distance between each ordered pair of points 
    from the input sets. The distance is computed between points with the same index:
    ppd(a[0], b[0])+ppd(a[1], b[1])+...

    point_set_a: ground truth
    point_set_b: prediction
    """
    with tf.name_scope(name):
        point_set_a, point_set_b = _fix_up_point_sets(point_set_a, point_set_b)
        dist = tf.reduce_sum(point_set_a * point_set_b, axis=-1) # (x_a*x_b) + (y_a*y_b) + (z_a*z_b) = a \dot b
        prod_abs = tf.math.sqrt(tf.reduce_sum(tf.square(point_set_a)))*tf.math.sqrt(tf.reduce_sum(tf.square(point_set_b)))
        dist = dist / prod_abs
        dist = tf.reduce_mean(dist, axis = -1) # mean for all points in a frame (avoids incomparable losses when a different number of markers is enabled)
        return dist

def selective_perpoint_cosine_distance(point_set_a, point_set_b, name="selective_cosine_dist"):
    """
    Apply Per-point cosine loss only to enabled markers

    point_set_a: ground truth
    point_set_b: prediction
    """
    point_set_a, point_set_b = _select_by_marker(point_set_a, point_set_b)
    return perpoint_cosine_distance(point_set_a, point_set_b, name)
