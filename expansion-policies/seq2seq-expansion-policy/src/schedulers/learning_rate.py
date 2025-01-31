from typing import Dict

import tensorflow as tf

@tf.keras.utils.register_keras_serializable()
class WarmupDecaySchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    A learning rate schedule that first linearly warms up from 0 to a peak
    (initial_lr) over `warmup_steps`, and then exponentially decays from that peak
    down to (final_decay_rate * initial_lr).

    Specifically:
      1. During the warmup phase (step < warmup_steps):
         lr = (step / warmup_steps) * initial_lr   [i.e. linear ramp]
      2. During the decay phase (step >= warmup_steps):
         we define a decay ratio in [0..1] that goes from 0 at step=warmup_steps to 1 at step=warmup_steps+decay_steps,
         then we do:
             lr = initial_lr * (final_decay_rate ^ decay_ratio)

    This allows a smooth ramp-up to `initial_lr`, then a smooth exponential
    descent to `initial_lr * final_decay_rate`.

    Example
    -------
    >>> schedule = WarmupDecaySchedule(
    ...     initial_lr=1e-4,
    ...     warmup_steps=5000,
    ...     decay_steps=40000,
    ...     final_decay_factor=0.1
    ... )
    >>> optimizer = tf.keras.optimizers.Adam(learning_rate=schedule)
    >>> model.compile(optimizer=optimizer, loss=..., metrics=...)
    """
    def __init__(
        self,
        initial_lr: float = 1e-4,
        warmup_steps: int = 5000,
        decay_steps: int = 40000,
        final_decay_factor: float = 0.1
    ):
        """
        Parameters
        ----------
        initial_lr : float
            The peak learning rate you want to reach by the end of warmup.
        warmup_steps : int
            Number of steps (batches) to do linear warmup from 0.0 -> initial_lr.
        decay_steps : int
            Number of steps over which we apply exponential decay after warmup.
        final_decay_factor : float
            The factor by which we reduce initial_lr after decay_steps.
            E.g., 0.1 means the LR is reduced to 10% of initial_lr by the end of decay.
        """
        super().__init__()
        self._initial_lr = initial_lr
        self._warmup_steps = warmup_steps
        self._decay_steps = decay_steps
        self._final_decay_factor = final_decay_factor

    def __call__(self, step):
        """
        Compute the learning rate (LR) at a given training step.

        1) If step < warmup_steps, current step is in the warmup phase:
           lr = (step / warmup_steps) * initial_lr
           This linearly increases from 0 to initial_lr as step goes from 0..warmup_steps.

        2) If step >= warmup_steps, current step is in the exponential decay phase:
           - Let post_warmup_steps = step - warmup_steps (clamped at min 0).
           - decay_ratio in [0, 1] => post_warmup_steps / decay_steps, but capped at 1.0
           - Then lr = initial_lr * (final_decay_rate ^ decay_ratio).

        Returns
        -------
        A scalar tf.Tensor representing the learning rate at 'step'.
        """
        # 1. Cast current optimiser step as float for math ops
        step_float = tf.cast(step, tf.float32)

        # 2. Warmup Phase:
        #    a) Ramp up learning rate from 0.0 to `initial_lr` over `warmup_steps`
        #    b) Defensive logic against `warmup_steps=0` with `tf.maximum(1.0, self.warmup_steps)`
        warmup_lr = (step_float / tf.maximum(1.0, self._warmup_steps)) * self._initial_lr

        # 3. Exponential Decay Phase:
        #    a) `post_warmup_steps` is how many steps past the `warmup_steps` threshold the current step is
        #    b) `decay_ratio` goes from 0 at `step = warmup_steps` (i.e. the end of the warmup phase) to 1 at
        #       `step = warmup_steps + decay_steps` (i.e. the end of the exponential decay phase)
        post_warmup_steps = tf.maximum(step_float - self._warmup_steps, 0.0)
        decay_ratio = tf.minimum(post_warmup_steps / self._decay_steps, 1.0)

        # Exponential decay from `initial_lr` down to `(final_decay_factor * initial_lr)`
        # Formula: lr = initial_lr * final_decay_factor^(decay_ratio)
        decayed_lr = self._initial_lr * (self._final_decay_factor ** decay_ratio)

        # 4. Conditional: If current step is in warmup phase (`< warmup_steps`), return `warmup_lr`;
        #    otherwise, use decayed_lr
        return tf.cond(step_float < self._warmup_steps, lambda: warmup_lr, lambda: decayed_lr)

    def get_config(self) -> Dict:
        """
        Returns the configuration of the scheduler for serialisation.

        This method enables the scheduler to be serialised and de-serialised with its
        configuration parameters, facilitating model saving and loading.

        Returns
        ----------
        config: dict
            A Python dictionary containing the scheduler's configuration.
        """
        return {
            "initial_lr": self._initial_lr,
            "warmup_steps": self._warmup_steps,
            "decay_steps": self._decay_steps,
            "final_decay_rate": self._final_decay_factor
        }


@tf.keras.utils.register_keras_serializable()
class VaswaniWarmupDecaySchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    A learning rate schedule based on the formula from the paper:
        "Attention Is All You Need" (Vaswani et al., 2017).

    Specifically, the schedule is:

        learning_rate(step) = (d_model ** -0.5) * min(step ** -0.5, step * (warmup_steps ** -1.5))

    Explanation:
    1. d_model is the dimensionality of embeddings.
    2. step is the current optimiser step (starting at 1, 2, 3, ...).
    3. warmup_steps is a hyperparameter (e.g., 4000). During early training, the term
       (step * warmup_steps^-1.5) dominates, causing the LR to increase linearly up to ~step=warmup_steps.
       After that, (step^-0.5) term dominates, and the LR decreases proportionally to the inverse square root of step.

    This schedule ensures a gentle ramp-up early (the so-called "warmup") and an inverse-square-root decay thereafter.

    Example
    -------
    >>> lr_schedule = VaswaniWarmupDecaySchedule(d_model=512, warmup_steps=4000)
    >>> optimizer = tf.keras.optimizers.Adam(
    ...     learning_rate=lr_schedule, beta_1=0.9, beta_2=0.98, epsilon=1e-9
    ... )
    >>> model.compile(optimizer=optimizer, loss=..., metrics=...)

    Reference:
        "Attention Is All You Need" (https://arxiv.org/abs/1706.03762),
        Section 5.3.
    """

    def __init__(self, d_model, warmup_steps=4000):
        super().__init__()
        self.d_model = d_model
        self.warmup_steps = float(warmup_steps)
        self.d_model_factor = d_model ** -0.5  # Precompute the scaling factor as d_model^-0.5:

    def __call__(self, step) -> tf.Tensor:
        """
        Compute the learning rate (LR) at a given training step.

        The formula:
            lr(step) = d_model^-0.5 * min(step^-0.5, step * warmup_steps^-1.5)

        1) 'step^-0.5' (inverse square root of the current step),
        2) 'step * warmup_steps^-1.5' (a linear warmup factor).

        During early steps, the second term is smaller, so the LR grows linearly.
        After 'warmup_steps', the first term becomes smaller, so the LR decays ~step^-0.5.

        Parameters
        ----------
        step : tf.Tensor
            The current optimiser step (scalar). Typically increments by 1 each mini-batch update.

        Returns
        -------
        tf.Tensor
            The scalar learning rate value for the current step.
        """
        # 1. Cast current optimiser step as float for math ops
        step_float = tf.cast(step, tf.float32)

        # 2. Calculate Term 1: `inverse_sqrt_step = step^-0.5`
        inverse_sqrt_step = tf.pow(step_float, -0.5)

        # 3. Calculate Term 2: `progressive_warmup = step * warmup_steps^-1.5 = step / (warmup_steps^(1.5))`
        progressive_warmup = step_float * tf.pow(self.warmup_steps, -1.5)

        # 4. Combine Term 1 & Term 2, take the minimum and multiply by `d_model^-0.5` to get and return current training
        #    step LR
        return self.d_model_factor * tf.minimum(inverse_sqrt_step, progressive_warmup)

    def get_config(self) -> Dict:
        """
        Returns the configuration of the scheduler for serialisation.

        This method enables the scheduler to be serialised and de-serialised with its
        configuration parameters, facilitating model saving and loading.

        Returns
        ----------
        config: dict
            A Python dictionary containing the scheduler's configuration.
        """
        return {
            "d_model": self.d_model,
            "warmup_steps": self.warmup_steps
        }