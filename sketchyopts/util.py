from typing import Callable, NamedTuple, Union, Mapping, Hashable

import jax

from optax._src import base
from optax._src import wrappers


def shareble_state_chain(
    *args: base.GradientTransformation,
) -> base.GradientTransformationExtraArgs:
  """Applies a list of chainable update transformations.

  Given a sequence of chainable transforms, `chain` returns an `init_fn`
  that constructs a `state` by concatenating the states of the individual
  transforms, and returns an `update_fn` which chains the update transformations
  feeding the appropriate state to each.

  Args:
    *args: a sequence of chainable (init_fn, update_fn) tuples.

  Returns:
    A ``GradientTransformationExtraArgs``, created by chaining the input
    transformations. Note that independent of the argument types, the resulting
    transformation always supports extra args. Any extra arguments passed to the
    returned transformation will be passed only to those transformations in the
    chain that support extra args.
  """

  transforms = [base.with_extra_args_support(t) for t in args]
  init_fns, update_fns = zip(*transforms)

  def init_fn(params):
    return tuple(fn(params) for fn in init_fns)

  def update_fn(updates, state, params=None, **extra_args):
    if len(update_fns) != len(state):
      raise ValueError('The number of updates and states has to be the same in '
                       'chain! Make sure you have called init first!')

    new_state = []
    for s, fn in zip(state, update_fns):
      updates, new_s = fn(updates, s, params, new_state, **extra_args)
      new_state.append(new_s)
    return updates, tuple(new_state)

  # We opt to always return the GradientTransformationExtraArgs type here,
  # instead of selecting the return type based on the arguments, since it works
  # much better with the currently available type checkers. It also means that
  # users will not get unexpected signature errors if they remove all of the
  # transformations in a chain accepting extra args.
  return base.GradientTransformationExtraArgs(init_fn, update_fn)


def shareble_state_named_chain(
    *transforms: tuple[str, base.GradientTransformation]
) -> base.GradientTransformationExtraArgs:
  """Chains optax gradient transformations.

  The `transforms` are `(name, transformation)` pairs, constituted of a string
  `name` and an associated gradient transformation `transformation`. The
  gradient transformation must be an instance of either `GradientTransformation`
  or `GradientTransformationExtraArgs`.

  Each `name` is used as key for the state of the corresponding transformation
  within the `named_chain` state. Thus the state of the gradient transformation
  with a given `name` can be retrieved as `opt_state[name]`.

  Example:

    # tx1 is a GradientTransformation with no extra_args. 
    # tx2 is a GradientTransformationExtraArgs that requires `loss`.
    # tx3 is a GradientTransformationExtraArgs that requires `temperature`.

    tx = named_chain(('one', tx1), ('two', tx2), ('three', tx3))
    extra_args={'loss': 0.3, 'temperature': 0.01}
    tx.init(params)
    tx.update(grads, state, params, **extra_args)

  Args:
    *transforms: an arbitrary number of `(name, tx)` pairs, constituted of a
      string `name` and an associated gradient transformation `tx`. The latter 
      is a `GradientTransformation` or `GradientTransformationExtraArgs`.

  Returns: 
    A single (init_fn, update_fn) tuple. 
  """

  names = [name for name, _ in transforms]

  if len(names) != len(set(names)):
    raise ValueError(
        f'Named transformations must have unique names, but got {names}')

  transforms = [
      (name, base.with_extra_args_support(t))
      for name, t in transforms]

  def init_fn(params):
    states = {}
    for (name, tx) in transforms:
      states[name] = tx.init(params)
    return states
  def update_fn(updates, state, params=None, **extra_args):
    new_state = {}
    for (name, tx) in transforms:
      updates, new_state[name] = tx.update(
          updates, state[name], params, new_state, **extra_args)
    return updates, new_state

  return base.GradientTransformationExtraArgs(init_fn, update_fn)


def scale_by_adaptive_learning_rate(
    learning_rate: base.ScalarOrSchedule,
    *,
    flip_sign: bool = True,
) -> base.GradientTransformation:
  """Scale by the (negative) learning rate (either as scalar or as schedule).

  Args:
    learning_rate: Can either be a scalar or a schedule (i.e. a callable that
      maps an (int) step to a float).
    flip_sign: When set to True (the default) this corresponds to scaling by the
      negative learning rate.

  Returns:
    An optax.GradientTransformation that corresponds to multiplying the gradient
    with `-learning_rate` (if flip_sign is True) or with `learning_rate` (if
    flip_sign is False).
  """
  m = -1 if flip_sign else 1
  if callable(learning_rate):
    return scale_by_schedule(lambda count: m * learning_rate(count))
  return scale(m * learning_rate)