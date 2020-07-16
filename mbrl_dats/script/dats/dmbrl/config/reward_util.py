from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf

import numpy as np

# The value returned by tolerance() at `margin` distance from `bounds` interval.
_DEFAULT_VALUE_AT_MARGIN = 0.1


def _sigmoids(x, value_at_1, sigmoid):
  """Returns 1 when `x` == 0, between 0 and 1 otherwise.
  Args:
    x: A scalar or numpy array.
    value_at_1: A float between 0 and 1 specifying the output when `x` == 1.
    sigmoid: String, choice of sigmoid type.
  Returns:
    A numpy array with values between 0.0 and 1.0.
  Raises:
    ValueError: If not 0 < `value_at_1` < 1, except for `linear`, `cosine` and
      `quadratic` sigmoids which allow `value_at_1` == 0.
    ValueError: If `sigmoid` is of an unknown type.
  """
  if sigmoid in ('cosine', 'linear', 'quadratic'):
    if not 0 <= value_at_1 < 1:
      raise ValueError('`value_at_1` must be nonnegative and smaller than 1, '
                       'got {}.'.format(value_at_1))
  else:
    if not 0 < value_at_1 < 1:
      raise ValueError('`value_at_1` must be strictly between 0 and 1, '
                       'got {}.'.format(value_at_1))

  if sigmoid == 'gaussian':
    scale = tf.sqrt(-2 * tf.log(value_at_1))
    return tf.exp(-0.5 * (x * scale) ** 2)

  elif sigmoid == 'hyperbolic':
    scale = tf.acosh(1 / value_at_1)
    return 1 / tf.cosh(x * scale)

  elif sigmoid == 'long_tail':
    scale = tf.sqrt(1 / value_at_1 - 1)
    return 1 / ((x * scale) ** 2 + 1)

  elif sigmoid == 'cosine':
    scale = tf.acos(2 * value_at_1 - 1) / np.pi
    scaled_x = x * scale
    return tf.where(abs(scaled_x) < 1,
                    (1 + tf.cos(np.pi * scaled_x)) / 2, 0.0 * scaled_x)

  elif sigmoid == 'linear':
    scale = 1.0 - value_at_1
    scaled_x = x * scale
    return tf.where(abs(scaled_x) < 1, 1 - scaled_x, 0.0 * scaled_x)

  elif sigmoid == 'quadratic':
    scale = tf.sqrt(1.0 - value_at_1)
    scaled_x = x * scale
    return tf.where(abs(scaled_x) < 1, 1 - scaled_x ** 2, 0.0 * scaled_x)

  elif sigmoid == 'tanh_squared':
    scale = tf.arctanh(tf.sqrt(1 - value_at_1))
    return 1 - tf.tanh(x * scale) ** 2

  else:
    raise ValueError('Unknown sigmoid type {!r}.'.format(sigmoid))


def tolerance(x, bounds=(0.0, 0.0), margin=0.0, sigmoid='gaussian',
              value_at_margin=_DEFAULT_VALUE_AT_MARGIN):
  """Returns 1 when `x` falls inside the bounds, between 0 and 1 otherwise.
  Args:
    x: A scalar or numpy array.
    bounds: A tuple of floats specifying inclusive `(lower, upper)` bounds for
      the target interval. These can be infinite if the interval is unbounded
      at one or both ends, or they can be equal to one another if the target
      value is exact.
    margin: Float. Parameter that controls how steeply the output decreases as
      `x` moves out-of-bounds.
      * If `margin == 0` then the output will be 0 for all values of `x`
        outside of `bounds`.
      * If `margin > 0` then the output will decrease sigmoidally with
        increasing distance from the nearest bound.
    sigmoid: String, choice of sigmoid type. Valid values are: 'gaussian',
       'linear', 'hyperbolic', 'long_tail', 'cosine', 'tanh_squared'.
    value_at_margin: A float between 0 and 1 specifying the output value when
      the distance from `x` to the nearest bound is equal to `margin`. Ignored
      if `margin == 0`.
  Returns:
    A float or numpy array with values between 0.0 and 1.0.
  Raises:
    ValueError: If `bounds[0] > bounds[1]`.
    ValueError: If `margin` is negative.
  """
  lower, upper = bounds
  if lower > upper:
    raise ValueError('Lower bound must be <= upper bound.')
  if margin < 0:
    raise ValueError('`margin` must be non-negative.')

  in_bounds = tf.logical_and(lower <= x, x <= upper)
  if margin == 0:
    value = tf.where(in_bounds, 1.0, 0.0)
  else:
    d = tf.where(x < lower, lower - x, x - upper) / margin
    value = tf.where(in_bounds,
                     1.0 + d * 0.0,
                     _sigmoids(d, value_at_margin, sigmoid))

  return value
