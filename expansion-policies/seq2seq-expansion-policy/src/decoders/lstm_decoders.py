from typing import List, Optional, Tuple, Union, Any

import tensorflow as tf
from tensorflow import Tensor
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import (Embedding, LSTM, Dropout,
                                     Dense, Layer, LayerNormalization)

from decoders.decoder_interface import DecoderInterface
from attention.attention import BahdanauAttention


@tf.keras.utils.register_keras_serializable()
class StackedLSTMDecoder(DecoderInterface):
    """
    StackedLSTMDecoder

    A custom TensorFlow Keras layer that generates the target sequence in a Seq2Seq model using the encoder's final
    context/state vectors and its own previously generated tokens. It includes:

    Architecture:
        - Embedding Layer: Transforms target token indices into dense embedding vectors.
        - Stacked LSTM Layers: Processes embeddings through multiple LSTM layers to capture sequential dependencies
            and patterns.
        - Dropout Layers: Applies dropout after each LSTM layer for regularization.
        - Layer Normalization Layers: Normalizes the outputs of each LSTM layer to stabilize and accelerate training.
        - Residual Connections: Implements residual (skip) connections from the embeddings to deeper LSTM layers to
            facilitate gradient flow and mitigate vanishing gradient issues.
        - Attention Mechanism: Utilizes a Bahdanau attention mechanism to focus on relevant parts of the encoder's
            outputs during decoding.
        - Projection Layers for Residual Connections: Transforms decoder outputs and context vectors before adding
            them together for residual connections around the attention mechanism.
        - Output Dense Layer: Produces probability distributions over the target vocabulary via softmax activation.

    Parameters
    ----------
    vocab_size : int
        Size of the target vocabulary.
    embedding_dim : int
        Dimensionality of the embedding vectors.
    units : int
        Number of units in each LSTM layer.
    num_layers : int, optional
        Number of stacked LSTM layers (default is 4).
    dropout_rate : float, optional
        Dropout rate applied after each LSTM layer (default is 0.2).
    weight_decay : float, optional
        L2 regularization factor applied to the LSTM and Dense layers (default is 1e-4).
    **kwargs : Any
        Additional keyword arguments passed to the base `Layer` class.

    Attributes
    ----------
    _embedding : Embedding
        Embedding layer that transforms target token indices into dense vectors.
    _lstm_layers : List[LSTM]
        List of stacked LSTM layers.
    _dropout_layers : List[Dropout]
        List of Dropout layers applied after each LSTM layer.
    _layer_norm_layers : List[LayerNormalization]
        List of Layer Normalization layers applied after each LSTM layer.
    _attention : BahdanauAttention
        Bahdanau attention mechanism instance.
    _decoder_dense : Dense
        Dense layer transforming decoder outputs for residual connections.
    _context_dense : Dense
        Dense layer transforming context vectors for residual connections.
    _output_layer_norm : LayerNormalization
        Layer Normalization applied after residual connections.
    _dense : Dense
        Output Dense layer that produces probability distributions over the target vocabulary.

    Methods
    -------
    call(inputs, training=False)
        Processes the input sequence, applies attention, and generates output probabilities over the target vocabulary.
    single_step(decoder_input, states, encoder_output)
        Performs a single decoding step for inference.
    compute_mask(inputs, mask=None)
        Computes the mask for the decoder input.
    get_config()
        Returns the configuration of the layer for serialization.
    from_config(config)
        Creates a StackedLSTMDecoder layer from its configuration.

    Returns
    -------
    decoder_output : tf.Tensor
        Predicted token probabilities for each timestep in the target sequence.
    """
    def __init__(
        self,
        vocab_size: int,
        decoder_embedding_dim: int,
        units: int,
        num_layers: int,
        attention_dim: int,
        dropout_rate: float = 0.2,
        weight_decay: float = 1e-4,
        supports_masking: Optional[bool] = True,
        **kwargs
    ) -> None:
        super(StackedLSTMDecoder, self).__init__(**kwargs)
        self._vocab_size = vocab_size
        self._embedding = Embedding(vocab_size, decoder_embedding_dim, mask_zero=True)
        self._units = units
        self._num_layers = num_layers
        self._attention_dim = attention_dim
        self._dropout_rate = dropout_rate
        self._weight_decay = weight_decay

        self._supports_masking = supports_masking

        # Build LSTM, Dropout, and LayerNormalization layers
        self._lstm_layers = []
        self._dropout_layers = []
        self._layer_norm_layers = []
        for i in range(num_layers):
            lstm_layer = LSTM(
                units=units,
                return_sequences=True,
                return_state=True,
                kernel_regularizer=l2(weight_decay) if weight_decay is not None else None,
                name=f'lstm_decoder_{i + 1}'
            )
            self._lstm_layers.append(lstm_layer)

            dropout_layer = Dropout(dropout_rate, name=f'decoder_dropout_{i + 1}')
            self._dropout_layers.append(dropout_layer)

            layer_norm_layer = LayerNormalization(name=f'decoder_layer_norm_{i + 1}')
            self._layer_norm_layers.append(layer_norm_layer)

        # Attention Mechanism
        self._attention: BahdanauAttention = BahdanauAttention(
            units=attention_dim
        )

        # Projection layers for residual connection
        self._decoder_dense = Dense(self._units, name='decoder_dense')
        self._context_dense = Dense(self._units, name='context_dense')
        self._output_layer_norm = LayerNormalization(name='output_layer_norm')

        # Output layer
        self._dense: Dense = Dense(
            vocab_size,
            activation='softmax',
            kernel_regularizer=l2(weight_decay) if weight_decay is not None else None
        )

    def call(
        self,
        inputs: Tuple[tf.Tensor, List[tf.Tensor], tf.Tensor],
        training: Optional[bool] = None,
        mask: Optional[tf.Tensor] = None
    ) -> tf.Tensor:
        """
        Processes the input sequence, applies attention, and generates output probabilities over the target vocabulary.

        This method performs the following steps:
            1. Embeds the decoder input tokens.
            2. Passes the embeddings through stacked LSTM layers with dropout and layer normalization.
            3. Applies residual connections to facilitate better gradient flow.
            4. Utilizes the Bahdanau attention mechanism to focus on relevant encoder outputs.
            5. Projects the decoder and context vectors for residual connections around the attention mechanism.
            6. Applies layer normalization and activation before generating the final output probabilities.

        Parameters
        ----------
        inputs : Tuple[tf.Tensor, List[tf.Tensor], tf.Tensor]
            A tuple containing:
                - decoder_input : tf.Tensor
                    Tensor of shape (batch_size, seq_len_dec) containing input token indices.
                - initial_state : List[tf.Tensor]
                    List of tensors representing the initial hidden and cell states for each LSTM layer.
                - encoder_output : tf.Tensor
                    Tensor of shape (batch_size, seq_len_enc, enc_units) containing encoder outputs.
        training : Optional[bool], default=None
            Training flag. If `True`, dropout layers are active. Defaults to `None`.
        mask : Optional[tf.Tensor], default=None
            Tuple containing decoder and encoder masks. If provided, should be a tuple `(decoder_mask, encoder_mask)`.

        Returns
        -------
        decoder_output : tf.Tensor
            The predicted token probabilities for each timestep in the target sequence.
            Shape: (batch_size, seq_len_dec, vocab_size).

        Raises
        ------
        ValueError
            If `decoder_input`, `initial_state`, or `encoder_output` are `None`.
            If the length of `initial_state` does not match the expected number based on `num_layers`.
        """
        # Unpack inputs
        decoder_input: tf.Tensor  # Shape: (batch_size, seq_len_dec)
        initial_state: List[tf.Tensor]  # List of tensors for initial hidden and cell states
        encoder_output: tf.Tensor  # Shape: (batch_size, seq_len_enc, enc_units)
        decoder_input, initial_state, encoder_output = inputs

        if decoder_input is None or initial_state is None or encoder_output is None:
            raise ValueError('decoder_input, initial_state and encoder_output must be passed to the Decoder.')

        # Unpack masks
        decoder_mask, encoder_mask = None, None
        if mask is not None:
            decoder_mask, encoder_mask = mask

        # Embed the input and extract decoder mask if not provided
        decoder_output = self.embedding(decoder_input) # Shape: (batch_size, seq_len_dec, decoder_embedding_dim)
        if decoder_mask is None:
            decoder_mask = self.embedding.compute_mask(decoder_input) # Shape: (batch_size, seq_len_dec)

        # Initialize previous_output with the embeddings
        previous_output = decoder_output

        # Prepare initial states for all layers
        num_states = len(initial_state)
        expected_states = self.num_layers * 2
        if num_states == 2:
            # Use initial_state (encoder final state) for the first LSTM layer
            states_list = [initial_state] + [(None, None)] * (self.num_layers - 1)
        elif num_states == expected_states:
            # Prepare initial states for all layers, initialising to zeros if not provided
            states_list = [
                (initial_state[i], initial_state[i + 1]) for i in range(0, num_states, 2)
            ]
        else:
            raise ValueError(f"Expected initial_state length to be 2 or {expected_states}, got {num_states}")

        # Process through decoder layers
        new_states: List[tf.Tensor] = []
        for i, (lstm_layer, dropout_layer, layer_norm_layer) in enumerate(
                zip(self._lstm_layers, self._dropout_layers, self._layer_norm_layers)
        ):
            state_h, state_c = states_list[i]
            if state_h is None or state_c is None:
                batch_size = tf.shape(decoder_output)[0]
                state_h = tf.zeros((batch_size, self._units))
                state_c = tf.zeros((batch_size, self._units))

            decoder_output, state_h, state_c = lstm_layer(
                decoder_output,
                mask=decoder_mask,
                initial_state=[state_h, state_c],
                training=training
            )
            new_states.extend([state_h, state_c])

            # Apply Layer Normalization
            decoder_output = layer_norm_layer(decoder_output)

            # Apply residual connection from the second layer onwards
            if i > 0:
                decoder_output += previous_output

            # Update previous_output
            previous_output = decoder_output

            # Apply dropout
            decoder_output = dropout_layer(decoder_output, training=training) # Shape: (batch_size, seq_len_dec, units)

        # Extract only the encoder_mask if passed mask list of tuple
        if mask is not None and isinstance(mask, (list, tuple)):
            encoder_mask = mask[1]
        else:
            encoder_mask = mask

        # Apply attention mechanism
        context_vector: tf.Tensor  # Shape: (batch_size, seq_len_dec, enc_units)
        attention_weights: tf.Tensor  # Shape: (batch_size, seq_len_dec, seq_len_enc)
        context_vector, attention_weights = self._attention(
            inputs=[encoder_output, decoder_output],
            mask=encoder_mask
        )

        # Transform decoder_output and context_vector for residual connections around attention mechanism
        decoder_transformed = self._decoder_dense(decoder_output)  # Shape: (batch_size, seq_len_dec, units)
        context_transformed = self._context_dense(context_vector)  # Shape: (batch_size, seq_len_dec, units)

        # Add transformed decoder outputs and context vector together for residual connection
        decoder_output = decoder_transformed + context_transformed # Shape: (batch_size, seq_len_dec, units)

        # Apply layer normalization and activation
        decoder_output = self._output_layer_norm(decoder_output)
        decoder_output = tf.nn.relu(decoder_output)

        # Generate output probabilities
        decoder_output: tf.Tensor = self._dense(decoder_output)

        return decoder_output

    def single_step(
        self,
        decoder_input: tf.Tensor,
        states: List[tf.Tensor],
        encoder_output: tf.Tensor
    ) -> tuple[Tensor, list[Tensor]]:
        """
        Performs a single decoding step.

        This method is typically used during inference to generate one token at a time.

        It processes the current decoder input, updates the LSTM states, applies attention,
        and produces the next token's probability distribution.

        Parameters
        ----------
        decoder_input : tf.Tensor
            The input token at the current time step with shape (batch_size, 1).
        states : List[tf.Tensor]
            The initial state of the decoder LSTM layers, containing hidden and cell states for each layer.
        encoder_output : tf.Tensor
            The output from the encoder with shape (batch_size, seq_len_enc, units).

        Returns
        -------
        Tuple[tf.Tensor, List[tf.Tensor]]
            - decoder_output : tf.Tensor
                Tensor of shape (batch_size, 1, vocab_size) containing the predicted token probabilities.
            - new_states : List[tf.Tensor]
                List of tensors representing the updated hidden and cell states for each LSTM layer.

        Raises
        ------
        ValueError
            If the length of `states` does not match the expected number based on `num_layers`.
        """
        # 1) Embed the current decoder input token
        decoder_output: tf.Tensor = self.embedding(decoder_input) # Shape: (batch_size, 1, decoder_embedding_dim)

        # 2) Verify the decoder has been initialised with the correct number of states for all layers
        expected_states = self.num_layers * 2 # 2 states per layer: hidden (h) and cell (c) states
        if len(states) != expected_states:
            raise ValueError(
                f"Expected {expected_states} states for {self.num_layers} layers, but got {len(states)}. "
                f"Ensure initialisation code creates a full set of states for all layers."
            )

        # 3) Group the flat list of states into (h, c) pairs per layer
        #    For example, states = [h1, c1, h2, c2, h3, c3, h4, c4] for 4 layers
        states_list = [
            (states[2 * i], states[2 * i + 1]) for i in range(self.num_layers)
        ]

        # Collect new h, c states in list
        new_states: List[tf.Tensor] = []

        # Store the input to the current layer for residual connections
        previous_output = decoder_output

        # 4) Process `decoder_input` through each stacked LSTM decoder layer
        for i, (lstm_layer, dropout_layer, layer_norm_layer) in enumerate(
                zip(self._lstm_layers, self._dropout_layers, self._layer_norm_layers)
        ):
            # Extract h, c states for this LSTM layer
            state_h, state_c = states_list[i]

            # Run one time step of this LSTM layer
            # `decoder_output` Shape: (batch_size, 1, units)
            # `state_h, state_c` Shape: (batch_size, units)
            decoder_output, new_h, new_c = lstm_layer(
                decoder_output,
                initial_state=[state_h, state_c],
                training=False
            )

            # Save updated states
            new_states.extend([new_h, new_c]) # Shape: (batch_size, 1, units)

            # Apply Layer Normalization
            decoder_output = layer_norm_layer(decoder_output)

            # Apply residual connection from the second layer onwards
            if i > 0:
                decoder_output += previous_output

            # Update previous_output
            previous_output = decoder_output

            # Apply dropout
            decoder_output = dropout_layer(decoder_output, training=False)

        # 5) Apply attention mechanism
        context_vector: tf.Tensor  # Shape: (batch_size, 1, enc_units)
        attention_weights: tf.Tensor  # Shape: (batch_size, 1, seq_len_enc)
        context_vector, attention_weights = self._attention(
            inputs=[encoder_output, decoder_output],
            mask=None  # No mask during inference
        )

        # 6) Apply the post-attention residual and normalisation
        #    Transform decoder_output and context_vector for residual connections around attention mechanism
        decoder_transformed = self._decoder_dense(decoder_output)  # Shape: (batch_size, 1, units)
        context_transformed = self._context_dense(context_vector)  # Shape: (batch_size, 1, units)

        # Add transformed decoder outputs and context vector together for residual connection
        decoder_output = decoder_transformed + context_transformed  # Shape: (batch_size, 1, units)

        # Apply layer normalization and ReLu activation
        decoder_output = self._output_layer_norm(decoder_output)
        decoder_output = tf.nn.relu(decoder_output)

        # 7) Generate output probability distributions
        decoder_output = self._dense(decoder_output)  # Shape: (batch_size, 1, vocab_size)

        return decoder_output, new_states

    def compute_mask(
        self,
        inputs: Union[tf.Tensor, List[tf.Tensor]],
        mask: Optional[Any] = None
    ) -> Optional[tf.Tensor]:
        """
        Computes the mask for the decoder input.

        This method propagates the mask forward by computing an output mask tensor
        based on the decoder's input mask. This ensures that the masking information
        is correctly passed to subsequent layers.

        Parameters
        ----------
        inputs : Union[tf.Tensor, List[tf.Tensor]]
            A tensor or list of tensors containing the decoder input, initial state, and encoder output.
        mask : Optional[Any], default=None
            Optional mask information. If provided, it should be a tuple containing
            the decoder and encoder masks.

        Returns
        -------
        decoder_mask: Optional[tf.Tensor]
            The mask tensor based on the decoder's input mask. Returns `None` if no mask is computed.
        """
        if isinstance(inputs, (list, tuple)) and len(inputs) == 3:
            decoder_input, _, _ = inputs
            decoder_mask = self.embedding.compute_mask(decoder_input)
            return decoder_mask
        return None

    def get_config(self):
        """
        Returns the configuration of the layer for serialisation.

        This method enables the layer to be serialised and de-serialised with its
        configuration parameters, facilitating model saving and loading.

        Returns
        ----------
        config: dict
            A Python dictionary containing the layer's configuration.
        """
        config = super().get_config()
        config.update({
            'vocab_size': self._vocab_size,
            'decoder_embedding_dim': self._embedding.output_dim,
            'units': self._units,
            'num_layers': self._num_layers,
            'attention_dim': self._attention_dim,
            'dropout_rate': self._dropout_rate,
            'weight_decay': self._weight_decay
        })
        return config

    @classmethod
    def from_config(cls, config: dict) -> 'StackedLSTMDecoder':
        """
        Creates a StackedLSTMDecoder layer from its configuration.

        This class method allows the creation of a new `StackedLSTMDecoder` instance
        from a configuration dictionary, enabling model reconstruction from saved configurations.

        Parameters
        ----------
        config : Dict[str, Any]
            A dictionary containing the configuration of the layer.

        Returns
        -------
        StackedLSTMDecoder
            A new instance of `StackedLSTMDecoder` configured using the provided config.
        """
        return cls(**config)

    @property
    def embedding(self) -> Embedding:
        """
        Returns the Embedding layer that transforms target token indices into dense vectors.

        Returns
        -------
        Embedding
            The decoders embedding layer.
        """
        return self._embedding

    @property
    def num_layers(self):
        """
        Returns the number of stacked LSTM layers in the decoder.

        Returns
        -------
        int
            Number of stacked LSTM layers (default is 4).
        """
        return self._num_layers
