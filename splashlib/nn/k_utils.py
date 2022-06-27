import tensorflow as tf

"""
Collection of utility functions.
"""

class StopExecution(Exception):
    def _render_traceback_(self):
        pass

def dict_to_str(di, indent=0):
    # convert a dictionary to a string with some nicer formatting
    strings = []
    for key, value in di.items():
        strings.append('\t' * indent + str(key))
        if isinstance(value, dict):
            strings.append(dict_to_str(value, indent+1))
        else:
            strings.append('\t' * (indent+1) + str(value))
    return '\n'.join(strings)

def isNotBlank(s):
    # s is not None and is not empty
    return bool(s and s.strip())

def isNoneOrBlank(s):
    # s is None or is empty
    return not isNotBlank(s)

def finite_reduce_mean(tensor, axis=-1, keepdims=False):
    # apply usual tf.math.reduce_mean only on finite values
    t_finite = tf.where(tf.math.is_finite(tensor), tensor, [0.0])
    t_size = tf.where(tf.math.is_finite(tensor), [1.0], [0.0])
    t_size = tf.math.reduce_sum(t_size, axis=axis, keepdims=keepdims)
    t_sum = tf.math.reduce_sum(t_finite, axis=axis, keepdims=keepdims)
    return t_sum/t_size

def finite_reduce_sum(tensor, axis=-1, keepdims=False):
    # apply usual tf.math.reduce_sum only on finite values
    t_finite = tf.where(tf.math.is_finite(tensor), tensor, [0.0])
    t_sum = tf.math.reduce_sum(t_finite, axis=axis, keepdims=keepdims)
    return t_sum