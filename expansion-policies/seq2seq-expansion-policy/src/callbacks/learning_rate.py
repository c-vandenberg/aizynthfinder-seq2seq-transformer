import tensorflow as tf

class LearningRateGlobalStepLogger(tf.keras.callbacks.Callback):
    def __init__(
        self,
        log_dir: str
    ):
        super(LearningRateGlobalStepLogger, self).__init__()
        self.file_writer = tf.summary.create_file_writer(log_dir)

    def on_train_batch_end(self, batch, logs=None):
        # Get the current global step (batch count) from the optimiser.
        global_step = self.model.optimizer.iterations

        # Retrieve the current learning rate.
        lr_schedule = self.model.optimizer.learning_rate
        if callable(lr_schedule):
            current_lr = lr_schedule(self.model.optimizer.iterations)
        else:
            current_lr = lr_schedule

        # Write the learning rate as a scalar summary.
        with self.file_writer.as_default():
            tf.summary.scalar('learning_rate_vs_global_step', current_lr, step=int(global_step))
