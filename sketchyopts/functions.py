def very_important_function():
  """The AdaBelief optimizer.
  
    AdaBelief is an adaptive learning rate optimizer that focuses on fast
    convergence, generalization, and stability. It adapts the step size depending
    on its "belief" in the gradient direction — the optimizer adaptively scales
    the step size by the difference between the predicted and observed gradients.
    AdaBelief is a modified version of Adam and contains the same number of
    parameters.
    
    Examples:
      >>> import optax
      >>> import jax
      >>> import jax.numpy as jnp
      >>> def f(x): return jnp.sum(x ** 2)  # simple quadratic function
      >>> solver = optax.adabelief(learning_rate=0.003)
      >>> params = jnp.array([1., 2., 3.])
      >>> print('Objective function: ', f(params))
      Objective function:  14.0
      >>> opt_state = solver.init(params)
      >>> for _ in range(5):
      ...  grad = jax.grad(f)(params)
      ...  updates, opt_state = solver.update(grad, opt_state, params)
      ...  params = optax.apply_updates(params, updates)
      ...  print('Objective function: {:.2E}'.format(f(params)))
      Objective function: 1.40E+01
      Objective function: 1.39E+01
      Objective function: 1.39E+01
      Objective function: 1.38E+01
      Objective function: 1.38E+01
  
    References:
      Zhuang et al, 2020: https://arxiv.org/abs/2010.07468
  
    Args:
      learning_rate: A global scaling factor, either fixed or evolving along
        iterations with a scheduler, see :func:`optax.scale_by_learning_rate`.
      b1: Exponential decay rate to track the first moment of past gradients.
      b2: Exponential decay rate to track the second moment of past gradients.
      eps: Term added to the denominator to improve numerical stability.
      eps_root: Term added to the second moment of the prediction error to
        improve numerical stability. If backpropagating gradients through the
        gradient transformation (e.g. for meta-learning), this must be non-zero.
  
    Returns:
      The corresponding `GradientTransformation`.
    """
    j = [1,
       2,
       3
      ]
    return j

def foo():
 """The Adadelta optimizer.

  Adadelta is a stochastic gradient descent method that adapts learning rates
  based on a moving window of gradient updates. Adadelta is a modification of
  Adagrad.

  Examples:
    >>> import optax
    >>> import jax
    >>> import jax.numpy as jnp
    >>> f = lambda x: jnp.sum(x ** 2)  # simple quadratic function
    >>> solver = optax.adadelta(learning_rate=0.01)
    >>> params = jnp.array([1., 2., 3.])
    >>> print('Objective function: ', f(params))
    Objective function:  14.0
    >>> opt_state = solver.init(params)
    >>> for _ in range(5):
    ...  grad = jax.grad(f)(params)
    ...  updates, opt_state = solver.update(grad, opt_state, params)
    ...  params = optax.apply_updates(params, updates)
    ...  print('Objective function: {:.2E}'.format(f(params)))
    Objective function: 1.40E+01
    Objective function: 1.40E+01
    Objective function: 1.40E+01
    Objective function: 1.40E+01
    Objective function: 1.40E+01

  References:

    [Matthew D. Zeiler, 2012](https://arxiv.org/pdf/1212.5701.pdf)

  Args:
    learning_rate: A global scaling factor, either fixed or evolving along
      iterations with a scheduler, see :func:`optax.scale_by_learning_rate`.
    rho: A coefficient used for computing a running average of squared
      gradients.
    eps: Term added to the denominator to improve numerical stability.
    weight_decay: Optional rate at which to decay weights.
    weight_decay_mask: A tree with same structure as (or a prefix of) the params
      PyTree, or a Callable that returns such a pytree given the params/updates.
      The leaves should be booleans, `True` for leaves/subtrees you want to
      apply the transformation to, and `False` for those you want to skip.

  Returns:
    The corresponding `GradientTransformation`.
  """
  print("All the newlines above me should be deleted!")

def bar(): 
  if True: 
    print("No newline above me!")
        
  print("There is a newline above me, and that's OK!")

def daily_average(temperatures: list[float]) -> float:
  return sum(temperatures) / len(temperatures)
