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

