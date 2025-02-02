import os
import json
import time
from typing import Dict, Any, List, Tuple, Union, Callable, Optional

import yaml
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    Callback,
    EarlyStopping,
    TensorBoard,
    ReduceLROnPlateau
)
from tensorflow.train import Checkpoint, CheckpointManager

from data.utils.logging_utils import configure_logger
from losses.losses import WeightedSparseCategoricalCrossEntropy
from metrics.smiles_string_metrics import SmilesStringMetrics
from schedulers.learning_rate import WarmupDecaySchedule, VaswaniWarmupDecaySchedule
from trainers.environment import TrainingEnvironment
from callbacks.checkpoints import BestValLossCallback
from callbacks.validation_metrics import ValidationMetricsCallback
from callbacks.gradient_monitoring import GradientMonitoringCallback
from metrics.perplexity import Perplexity
from data.utils.data_loader import DataLoader
from data.utils.tokenisation import SmilesTokeniser
from data.utils.preprocessing import TokenisedSmilesPreprocessor
from data.utils.logging_utils import (compute_metrics, log_metrics, print_metrics,
                                log_sample_predictions, print_sample_predictions)
from models.seq2seq import RetrosynthesisSeq2SeqModel
from models.utils import Seq2SeqModelUtils


class Trainer:
    """
    Trainer class for training and evaluating the Retrosynthesis Seq2Seq model.

    This class handles the setup of the environment, data loading, model
    initialisation, training, evaluation, and saving of the model.
    """
    def __init__(self, config_path: str) -> None:
        """
        Initializes the Trainer with configurations.

        Parameters
        ----------
        config_path : str
            Path to the configuration YAML file.

        Returns
        -------
        None
        """
        self._config:Dict[str, Any] = self._load_config(config_path)

        self._tokeniser: Optional[SmilesTokeniser] = None
        self._data_loader: Optional[DataLoader] = None
        self._vocab_size: Optional[int] = None
        self._encoder_preprocessor: Optional[TokenisedSmilesPreprocessor] = None
        self._decoder_preprocessor: Optional[TokenisedSmilesPreprocessor] = None
        self._model: Optional[RetrosynthesisSeq2SeqModel] = None
        self._optimizer: Optional[Adam] = None
        self._loss_function: Optional[Any] = None
        self._metrics: Optional[List[str]] = None
        self._callbacks: Optional[List[Callback]] = None

        os.makedirs(os.path.dirname(
            self._config.get('data', {}).get('logger_path', 'var/log/default_logs.log')),
            exist_ok=True
        )
        self._logger = configure_logger(self._config.get('data', {}).get('logger_path', 'var/log/default_logs.log'))

        self._initialize_components()

    @staticmethod
    def _load_config(config_path: str) -> dict:
        """
        Loads configuration from a YAML file.

        Parameters
        ----------
        config_path : str
            Path to the YAML configuration file.

        Returns
        -------
        Dict[str, Any]
            Configuration dictionary.

        Raises
        ------
        FileNotFoundError
            If the configuration file does not exist.
        yaml.YAMLError
            If there is an error parsing the YAML file.
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at: {config_path}")

        with open(config_path, 'r') as file:
            try:
                config: Dict[str, Any] = yaml.safe_load(file)
            except yaml.YAMLError as e:
                raise yaml.YAMLError(f"Error parsing YAML file: {e}")
        return config

    def _initialize_components(self) -> None:
        """
        Initialize DataLoader, Tokenizer, Preprocessor, and hyperparameters.

        Returns
        -------
        None

        Raises
        ------
        KeyError
            If required configuration keys are missing.
        """
        # Retrieve configurations
        data_conf: Dict[str, Any] = self._config.get('data', {})
        train_conf: Dict[str, Any] = self._config.get('training', {})

        # Initialize DataLoader
        self._data_loader = DataLoader(
            products_file=data_conf.get('products_file', ''),
            reactants_file=data_conf.get('reactants_file', ''),
            test_split=data_conf.get('test_split', 0.1),
            validation_split=data_conf.get('validation_split', 0.1),
            logger=self._logger,
            num_samples=train_conf.get('num_samples'),
            max_encoder_seq_length=data_conf.get('max_encoder_seq_length', 140),
            max_decoder_seq_length=data_conf.get('max_decoder_seq_length', 140),
            batch_size=data_conf.get('batch_size', 16),
            random_state=data_conf.get('random_state', 42),
            max_tokens = data_conf.get('max_tokens', 150),
            reverse_input_sequence=train_conf.get('reverse_tokenized_input_sequence', True)
        )

        # Load and prepare data
        self._data_loader.load_and_prepare_data()

        # Access tokenizer and vocab size
        self._tokeniser = self._data_loader.smiles_tokeniser
        self._vocab_size = self._data_loader.vocab_size

        self._logger.info(f"Tokeniser Vocabulary Size: {self._vocab_size}")

        # Save the tokenizer
        self._save_tokenizer(data_conf.get('tokenizer_save_path', 'tokenizer.json'))

        # Initialize Preprocessors
        self._encoder_preprocessor = TokenisedSmilesPreprocessor(
            smiles_tokenizer=self._data_loader.smiles_tokeniser,
            max_seq_length=data_conf.get('max_encoder_seq_length', 140)
        )
        self._decoder_preprocessor = TokenisedSmilesPreprocessor(
            smiles_tokenizer=self._data_loader.smiles_tokeniser,
            max_seq_length=data_conf.get('max_decoder_seq_length', 140)
        )

    def _save_tokenizer(self, tokenizer_path: str) -> None:
        """
        Saves the tokenizer's vocabulary to a JSON file.

        Parameters
        ----------
        tokenizer_path : str
            Path where the tokenizer vocabulary JSON will be saved.

        Returns
        -------
        None

        Raises
        ------
        IOError
            If the tokenizer cannot be saved to the specified path.
        """
        os.makedirs(os.path.dirname(tokenizer_path), exist_ok=True)
        try:
            with open(tokenizer_path, 'w') as f:
                json.dump(self._tokeniser.word_index, f, indent=4)
            self._logger.info(f"Tokenizer vocabulary saved to {tokenizer_path}")
        except IOError as e:
            self._logger.error(f"Failed to save tokenizer to {tokenizer_path}: {e}")
            raise

    def _setup_model(self) -> None:
        """
        Initializes and compiles the Seq2Seq model.

        Returns
        -------
        None

        Raises
        ------
        KeyError
            If required model configuration keys are missing.
        """
        model_conf: Dict[str, Any] = self._config.get('model', {})
        data_conf: Dict[str, Any] = self._config.get('data', {})
        train_conf: Dict[str, Any] = self._config.get('training', {})

        # Retrieve model and training parameters with defaults
        encoder_embedding_dim: int = model_conf.get('encoder_embedding_dim', 256)
        decoder_embedding_dim: int = model_conf.get('decoder_embedding_dim', 256)
        units: int = model_conf.get('units', 256)
        attention_dim: int = model_conf.get('attention_dim', 256)
        encoder_num_layers: int = model_conf.get('encoder_num_layers', 2)
        decoder_num_layers: int = model_conf.get('decoder_num_layers', 4)
        dropout_rate: float = train_conf.get('dropout_rate', 0.2)
        weight_decay: Union[float, None] = train_conf.get('weight_decay', None)
        lr_scheduler: str = train_conf.get('lr_scheduler', 'reduce_on_plateau')
        initial_lr: float = float(train_conf.get('initial_lr', 1e-4))
        lr_scheduler_confs: Dict = train_conf.get('lr_scheduler_configs', {})

        # Initialise the model
        self._model: RetrosynthesisSeq2SeqModel = RetrosynthesisSeq2SeqModel(
            input_vocab_size=self._vocab_size,
            output_vocab_size=self._vocab_size,
            encoder_embedding_dim=encoder_embedding_dim,
            decoder_embedding_dim=decoder_embedding_dim,
            attention_dim=attention_dim,
            smiles_tokenizer=self._tokeniser,
            units=units,
            num_encoder_layers=encoder_num_layers,
            num_decoder_layers=decoder_num_layers,
            dropout_rate=dropout_rate,
            weight_decay=weight_decay
        )

        if lr_scheduler == 'warmup_then_decay':
            self._logger.info('Optimiser using "warm up then decay" learning rate scheduler.')
            lr_warmup_decay_conf: Dict = lr_scheduler_confs.get('warmup_then_decay', {})
            total_steps: int = self._data_loader.train_steps_per_epoch * train_conf.get('epochs', 10)
            lr_warmup_ratio: float = lr_warmup_decay_conf.get('lr_warmup_ratio', 0.05)
            lr_decay_ratio: float = lr_warmup_decay_conf.get('lr_decay_ratio', 0.80)
            lr_decay_factor: float = lr_warmup_decay_conf.get('lr_decay_factor', 0.1)

            ##learning_rate = VaswaniLRSchedule(
            ##    d_model=encoder_embedding_dim,
            ##    warmup_steps=float(total_steps * lr_warmup_ratio)
            ##)

            learning_rate = WarmupDecaySchedule(
                initial_lr=initial_lr,
                warmup_steps=int(total_steps * lr_warmup_ratio),
                decay_steps=int(total_steps * lr_decay_ratio),
                final_decay_factor=lr_decay_factor
            )
        elif lr_scheduler == 'cyclical':
            self._logger.info('Optimiser using cyclical learning rate scheduler.')
            lr_cyclical_conf: Dict = lr_scheduler_confs.get('cyclical', {})
            maximal_lr: float = float(lr_cyclical_conf.get('maximal_lr', 1e-3))
            num_half_cycles_per_epoch: int = int(2 * lr_cyclical_conf.get('num_cycles_per_epoch', 1))

            def cyclical_lr_scheduler_decay_amp_scale_fn(cycle_index):
                return 1.0 / (2.0 ** (cycle_index - 1))

            learning_rate = tfa.optimizers.CyclicalLearningRate(
                initial_learning_rate=initial_lr,
                maximal_learning_rate=maximal_lr,
                scale_fn=cyclical_lr_scheduler_decay_amp_scale_fn,
                step_size=round(self._data_loader.train_steps_per_epoch / num_half_cycles_per_epoch)
            )

        else:
            self._logger.info('Optimiser using static learning rate.')
            learning_rate = initial_lr

        # Set up the optimiser
        self._optimizer: Adam = Adam(learning_rate=learning_rate, clipnorm=5.0)

        # Set up the loss function and metrics.
        # For loss:
        #   - If `use_weighted_loss` config is `True`, we build a token-to-weight map where a tokens weight is
        #     dictated by its frequency in the training dataset. These weights are then applied to each token class in
        #     a custom `WeightedSparseCategoricalCrossEntropy`.
        #   - If `use_weighted_loss` config is `False`, we simply use the core
        #     `tf.keras.losses.SparseCategoricalCrossentropy()` method.
        #
        # For accuracy:
        #   - Because our sequences are integer-encoded, we have to specify `SparseCategoricalAccuracy`
        #     (Kera's default accuracy is CategoricalAccuracy, which is for sequences that are one-hot encoded)
        use_weighted_loss: bool = data_conf.get('use_weighted_loss', False)
        if use_weighted_loss:
            self._logger.info(f"Using custom `WeightedSparseCategoricalCrossEntropy` loss function ")
            token_to_weight_map = self._tokeniser.build_token_weight_map(
                token_counts=self._tokeniser.token_counts
            )

            self._loss_function = WeightedSparseCategoricalCrossEntropy(
                token_to_weight_map=token_to_weight_map,
                from_logits=False
            )
        else:
            self._logger.info(f"Using core `tf.keras.losses.SparseCategoricalCrossentropy` loss function ")
            self._loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

        self._metrics = [
            tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            Perplexity(loss_function=self._loss_function)
        ]

        # Compile the model
        self._model.compile(
            optimizer=self._optimizer,
            loss=self._loss_function,
            metrics=self._metrics
        )

    def _build_model(self) -> None:
        """
        Builds the model by running a sample input through it to initialize weights.

        Returns
        -------
        None

        Raises
        ------
        StopIteration
            If the training dataset is empty.
        """
        self._logger.info("Building the model with sample data to initialize variables...")

        # Get a batch from the training dataset
        train_dataset = self._data_loader.get_train_dataset()
        try:
            sample_batch = next(iter(train_dataset))
        except StopIteration:
            raise StopIteration("Training dataset is empty. Cannot build the model.")

        (sample_encoder_input, sample_decoder_input), _ = sample_batch

        # Run the model on sample data
        self._model([sample_encoder_input, sample_decoder_input])

        self._logger.info("Model built successfully.\n")

    def _setup_callbacks(self) -> None:
        """
        Sets up training callbacks including EarlyStopping, TensorBoard, Checkpointing, and Learning Rate Scheduler.

        Returns
        -------
        None

        Raises
        ------
        KeyError
            If required training configuration keys are missing.
        """
        train_conf: Dict[str, Any] = self._config.get('training', {})
        lr_scheduler_confs: Dict = train_conf.get('lr_scheduler_configs', {})

        # Early Stopping
        early_stopping: EarlyStopping = EarlyStopping(
            monitor='val_loss',
            patience=train_conf.get('patience', 5),
            restore_best_weights=True
        )

        # Checkpoint manager
        checkpoint_dir = train_conf.get('checkpoint_dir', './checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint: Checkpoint = Checkpoint(model=self._model, optimizer=self._optimizer)
        checkpoint_manager: CheckpointManager = CheckpointManager(
            checkpoint,
            directory=checkpoint_dir,
            max_to_keep=5
        )

        # Restore from latest checkpoint if exists
        if checkpoint_manager.latest_checkpoint:
            checkpoint.restore(checkpoint_manager.latest_checkpoint)
            self._logger.info(f"Restored from {checkpoint_manager.latest_checkpoint}")
        else:
            self._logger.info("Initializing from scratch.")

        # Checkpoint Callback
        best_val_loss_checkpoint_callback: BestValLossCallback = BestValLossCallback(
            checkpoint_manager
        )

        # Validation metrics callback
        valid_metrics_dir: str = train_conf.get('valid_metrics_dir', './validation-metrics')
        tensorboard_dir: str = train_conf.get('tensorboard_dir', './tensorboard')
        validation_metrics_callback: ValidationMetricsCallback = ValidationMetricsCallback(
            tokenizer=self._tokeniser,
            validation_data=self._data_loader.get_valid_dataset(),
            validation_metrics_dir=valid_metrics_dir,
            tensorboard_dir=os.path.join(tensorboard_dir, 'validation_metrics'),
            logger=self._logger,
            max_length=self._data_loader.max_decoder_seq_length
        )

        # TensorBoard
        tensorboard_callback: TensorBoard = TensorBoard(
            log_dir=tensorboard_dir
        )

        # Gradient monitoring
        gradient_callback = GradientMonitoringCallback(
            log_dir=os.path.join(tensorboard_dir, 'gradients')
        )

        self._callbacks = [
            early_stopping,
            best_val_loss_checkpoint_callback,
            validation_metrics_callback,
            tensorboard_callback
        ]

        lr_scheduler: str = train_conf.get('lr_scheduler', 'reduce_on_plateau')

        if lr_scheduler == 'reduce_on_plateau':
            lr_reduce_on_plateau_conf: Dict = lr_scheduler_confs.get('reduce_on_plateau', {})
            lr_reduction_factor: float = lr_reduce_on_plateau_conf.get('lr_reduction_factor', 0.1)
            lr_reduce_patience: int = lr_reduce_on_plateau_conf.get('lr_reduce_patience', 3)

            # Reduce on plateau learning Rate Scheduler
            lr_scheduler: ReduceLROnPlateau = ReduceLROnPlateau(
                monitor='val_loss',
                factor=lr_reduction_factor,
                patience=lr_reduce_patience
            )

            self._callbacks.append(lr_scheduler)

    def _train(self) -> None:
        """
        Trains the Seq2Seq model using the training and validation datasets.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If dataloader or training/validation datasets are not initialised.
        """
        training_conf: Dict[str, Any] = self._config.get('training', {})

        if self._data_loader is None:
            raise ValueError("DataLoader is not initialised.")

        train_dataset = self._data_loader.get_train_dataset()
        valid_dataset = self._data_loader.get_valid_dataset()

        if train_dataset is None or valid_dataset is None:
            raise ValueError("Training or validation datasets are not available.")

        self._model.fit(
            train_dataset,
            epochs=training_conf.get('epochs', 10),
            validation_data=valid_dataset,
            callbacks=self._callbacks
        )

    def _evaluate(self) -> None:
        """
        Evaluates the trained model on the test dataset.

        Computes various metrics including loss, accuracy, perplexity, BLEU score, chemical validity,
        Tanimoto similarity, and Levenshtein distance. Results are logged and printed.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If tokenizer is not set or if required metrics functions are not available.
        """
        if self._tokeniser is None:
            raise ValueError("Tokenizer is not initialized.")

        training_conf: Dict[str, Any] = self._config.get('training', {})
        model_conf: Dict[str, Any] = self._config.get('model', {})

        test_dataset = self._data_loader.get_test_dataset()

        # Get test subset fraction for partial test evaluation
        test_subset_fraction: float = training_conf.get('test_subset_fraction', 1.0)

        if test_subset_fraction < 0.0 or test_subset_fraction > 1.0:
            raise ValueError(f"Test subset fraction must be a non-negative float under 1.0: "
                             f"{test_subset_fraction} found")

        if test_subset_fraction < 1.0:
            test_dataset_size: int = self._data_loader.test_size
            partial_count: int = int(test_dataset_size * test_subset_fraction)

            # Shuffle and take the first `partial_count` samples
            test_dataset = (
                test_dataset
                .shuffle(buffer_size=test_dataset_size, seed=self._data_loader.random_state)
                .take(partial_count)
            )

            self._logger.info(
                f"Subsampling test dataset to {partial_count} out of {test_dataset_size} "
                f"({test_subset_fraction * 100:.2f}%)."
            )

        test_metrics_dir: str = training_conf.get('test_metrics_dir', './evaluation')

        # Evaluate the model on the test dataset
        evaluation_results: List[float] = self._model.evaluate(test_dataset)
        if len(evaluation_results) < 3:
            raise ValueError("Expected at least three evaluation metrics (loss, accuracy, perplexity).")

        test_loss, test_accuracy, test_perplexity = evaluation_results[:3]

        references: List[List[List[str]]] = []
        hypotheses: List[List[str]] = []
        target_smiles: List[str] = []
        predicted_smiles: List[str] = []
        start_token: str = self._tokeniser.start_token
        end_token: str = self._tokeniser.end_token

        beam_search_start_time: float = time.time()
        for (encoder_input, decoder_input), target_sequences in test_dataset:
            # Generate sequences
            predicted_sequences_list, predicted_scores_list = self._model.predict_sequence_beam_search(
                encoder_input=encoder_input,
                beam_width=model_conf.get('beam_width', 5),
                max_length=self._data_loader.max_decoder_seq_length,
                start_token_id=self._tokeniser.word_index.get(start_token),
                end_token_id=self._tokeniser.word_index.get(end_token),
                return_top_n=1
            )

            top_predicted_sequences = [seq_list[0] for seq_list in predicted_sequences_list]

            # Convert sequences to text
            predicted_texts: List[str]  = self._tokeniser.sequences_to_texts(
                top_predicted_sequences,
                is_input_sequence=False
            )
            target_texts: List[str]  = self._tokeniser.sequences_to_texts(
                target_sequences,
                is_input_sequence=False
            )

            for ref, hyp in zip(target_texts, predicted_texts):
                ref_tokens: List[str]  = ref.split()
                hyp_tokens: List[str]  = hyp.split()
                references.append([ref_tokens])
                hypotheses.append(hyp_tokens)
                target_smiles.append(ref)
                predicted_smiles.append(hyp)

        beam_search_end_time: float = time.time()
        beam_search_time = beam_search_end_time - beam_search_start_time
        self._logger.info(f'Test Dataset Beam Search Time: {round(beam_search_time)} seconds')

        testing_metrics_start_time: float = time.time()
        metrics: Dict[str, float] = {
            'Test Loss': test_loss,
            'Test Accuracy': test_accuracy,
            'Test Perplexity': test_perplexity,
        }

        additional_metrics: Dict[str, float] = compute_metrics(
            references=references,
            hypotheses=hypotheses,
            target_smiles=target_smiles,
            predicted_smiles=predicted_smiles,
            evaluation_stage='Test',
            smiles_string_metrics=SmilesStringMetrics()
        )

        metrics.update(additional_metrics)

        log_metrics(
            metrics=metrics,
            directory=test_metrics_dir,
            filename='test_metrics.txt',
            separator='-' * 40
        )

        print_metrics(logger=self._logger, metrics=metrics)

        log_sample_predictions(
            target_smiles=target_smiles,
            predicted_smiles=predicted_smiles,
            directory=test_metrics_dir,
            filename='test_sample_predictions.txt',
            num_samples=5,
            separator_length=153
        )

        print_sample_predictions(
            logger=self._logger,
            target_smiles=target_smiles,
            predicted_smiles=predicted_smiles,
            num_samples=5,
            separator_length=153
        )

        testing_metrics_end_time: float = time.time()
        testing_metrics_time = testing_metrics_end_time - testing_metrics_start_time
        self._logger.info(f'Testing Metrics Time: {round(testing_metrics_time)} seconds')

    def _save_model(self) -> None:
        """
        Saves the trained model in TensorFlow SavedModel format and ONNX format.

        The method inspects the model layers, saves the model in Keras, HDF5, ONNX, and SavedModel formats.
        It ensures that the model is saved in multiple formats for compatibility and deployment purposes.

        Returns
        -------
        None

        Raises
        ------
        Exception
            If any of the model saving processes fail.
        """
        Seq2SeqModelUtils.inspect_model_layers(model=self._model)
        training_conf: Dict[str, Any] = self._config.get('training', {})
        data_conf: Dict[str, Any] = self._config.get('data', {})
        model_save_dir: str = training_conf.get('model_save_dir', './model')
        keras_save_dir: str = os.path.join(model_save_dir, 'keras')
        hdf5_save_dir: str = os.path.join(model_save_dir, 'hdf5')
        onnx_save_dir: str = os.path.join(model_save_dir, 'onnx')
        saved_model_save_dir: str = os.path.join(model_save_dir, 'saved_model')

        try:
            Seq2SeqModelUtils.model_save_keras_format(
                keras_save_dir=keras_save_dir,
                model=self._model
            )
        except Exception as e:
            self._logger.error(f"Error saving model in Keras format: {e}")

        try:
            Seq2SeqModelUtils.model_save_hdf5_format(
                hdf5_save_dir=hdf5_save_dir,
                model=self._model
            )
        except Exception as e:
            self._logger.error(f"Error saving model in HDF5 format: {e}")

        try:
            Seq2SeqModelUtils.model_save_onnx_format(
                onnx_output_dir=onnx_save_dir,
                model=self._model,
                max_encoder_seq_length=data_conf.get('max_encoder_seq_length', 140),
                max_decoder_seq_length=data_conf.get('max_decoder_seq_length', 140)
            )
        except Exception as e:
            self._logger.error(f"Error saving model in ONNX format: {e}")

        try:
            Seq2SeqModelUtils.save_saved_model_format(
                model_save_path=saved_model_save_dir,
                model=self._model
            )
        except Exception as e:
            self._logger.error(f"Error saving model in SavedModel format: {e}")

    def run(self):
        """
        Executes the full training pipeline.

        The pipeline includes:
        1. Setting up the training environment.
        2. Setting up and compiling the model.
        3. Building the model by initializing weights.
        4. Setting up training callbacks.
        5. Training the model.
        6. Saving the trained model.
        7. Evaluating the model on the test dataset.

        Returns
        -------
        None

        Raises
        ------
        Exception
            If any step in the training pipeline fails.
        """
        try:
            TrainingEnvironment.setup_environment(self._config)
            self._setup_model()
            self._build_model()
            self._setup_callbacks()
            self._train()
            self._model.summary()
            self._save_model()
            self._evaluate()
        except Exception as e:
            self._logger.error(f"An error occurred during the training pipeline: {e}")
            raise

def custom_train_step(
    model: tf.keras.Model,
    optimizer: tf.keras.optimizers.Optimizer,
    loss_fn: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
    gradient_callback: Any
) -> Callable[[Tuple[tf.Tensor, tf.Tensor]], tf.Tensor]:
    """
    Creates a custom training step function.

    This function defines a custom training step that computes the loss, calculates gradients,
    applies them using the optimizer, and invokes a gradient callback for monitoring or logging.

    Parameters
    ----------
    model : tf.keras.Model
        The Keras model to be trained.
    optimizer : tf.keras.optimizers.Optimizer
        The optimizer used to apply gradients.
    loss_fn : Callable[[tf.Tensor, tf.Tensor], tf.Tensor]
        A function that computes the loss given true and predicted values.
    gradient_callback : Any
        An object that has an `on_gradients_computed` method to handle gradient-related operations
        (e.g., logging, monitoring).

    Returns
    -------
    Callable[[Tuple[tf.Tensor, tf.Tensor]], tf.Tensor]
        A TensorFlow function that performs a single training step.

    Raises
    ------
    AttributeError
        If `gradient_callback` does not have an `on_gradients_computed` method.
    """
    if not hasattr(gradient_callback, 'on_gradients_computed'):
        raise AttributeError("gradient_callback must have an 'on_gradients_computed' method.")

    @tf.function
    def train_step(inputs: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        """
        Performs a single training step.

        Parameters
        ----------
        inputs : Tuple[tf.Tensor, tf.Tensor]
            A tuple containing input features and target labels:
            - inputs[0]: tf.Tensor representing input features.
            - inputs[1]: tf.Tensor representing target labels.

        Returns
        -------
        tf.Tensor
            The computed loss for the training step.
        """
        x, y = inputs
        with tf.GradientTape() as tape:
            y_pred = model(x, training=True)
            loss = loss_fn(y, y_pred)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        # Call gradient callback
        gradient_callback.on_gradients_computed(gradients, model.trainable_variables)

        return loss

    return train_step
