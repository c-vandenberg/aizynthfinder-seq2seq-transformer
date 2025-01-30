import tensorflow as tf

@tf.keras.utils.register_keras_serializable()
class WarmupThenDecaySchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(
        self,
        initial_lr: float = 1e-4,
        warmup_steps: int = 5000,
        decay_steps: int = 40000,
        final_decay_rate: float = 0.1
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
        final_decay_rate : float
            The factor by which we reduce initial_lr after decay_steps.
            E.g., 0.1 means the LR is reduced to 10% of initial_lr by the end of decay.
        """
        super().__init__()
        self.initial_lr = initial_lr
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.final_decay_rate = final_decay_rate

    def __call__(self, step):
        """Returns the learning rate as a function of the current step."""
        # Ensure we treat step as float
        step = tf.cast(step, tf.float32)

        # 1. Warmup Phase: linear ramp up from 0.0 to initial_lr
        #    over [0 .. warmup_steps] steps
        warmup_lr = (step / tf.maximum(1.0, self.warmup_steps)) * self.initial_lr

        # 2. Exponential Decay Phase: after warmup
        #    We'll map step=warmup_steps to ratio=0, step=warmup_steps+decay_steps to ratio=1
        post_warmup_steps = tf.maximum(step - self.warmup_steps, 0.0)
        decay_ratio = tf.minimum(post_warmup_steps / self.decay_steps, 1.0)
        # Exponential decay from initial_lr down to (final_decay_rate * initial_lr)
        # formula: lr = initial_lr * final_decay_rate^(decay_ratio)
        decayed_lr = self.initial_lr * (self.final_decay_rate ** decay_ratio)

        # pick whichever phase we are in
        lr = tf.cond(step < self.warmup_steps, lambda: warmup_lr, lambda: decayed_lr)
        return lr

    def get_config(self):
        return {
            "initial_lr": self.initial_lr,
            "warmup_steps": self.warmup_steps,
            "decay_steps": self.decay_steps,
            "final_decay_rate": self.final_decay_rate
        }
