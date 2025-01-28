import logging
import json
import math
from collections import Counter
from typing import Dict, List, Union

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
from deepchem.feat.smiles_tokenizer import BasicSmilesTokenizer

class SmilesTokeniser:
    """
    Tokenizer for SMILES strings.

    This tokenizer converts SMILES strings into individual character tokens,
    adds optional start and end tokens, and uses a Keras Tokenizer internally
    to convert tokens to indices.

    Attributes
    ----------
    start_token : str, optional
        Token representing the start of a sequence (default is '<START>').
    end_token : str, optional
        Token representing the end of a sequence (default is '<END>').
    oov_token : str, optional
        Token representing out-of-vocabulary tokens (default is '').
    max_tokens : int, optional
        Maximum number of tokens in the vocabulary (default is 150).
    _reverse_input_sequence : bool, optional
        Whether to reverse the input sequence during tokenization (default is False).

    Methods
    -------
    from_json(tokenizer_path: str, reverse_input_sequence: bool = False, start_token: str = '<START>',
              end_token: str = '<END>', oov_token: str = '') -> 'SmilesTokenizer'
        Class method to load a tokenizer from a JSON file with defensive checks.
    tokenize(smiles: str, is_input_sequence: bool) -> str:
        Tokenizes a single SMILES string into individual characters.
    tokenize_list(smiles_list: List[str], is_input_sequence = False) -> List[str]:
        Tokenizes a list of SMILES strings.
    adapt(tokenized_smiles_list: List[str]) -> None:
        Adapts the TextVectorization layer to the preprocessed SMILES list.
    texts_to_sequences(texts: List[str]) -> tf.Tensor:
        Converts tokenized texts to sequences of token indices.
    sequences_to_texts(sequences: Union[tf.Tensor, np.ndarray], is_input_sequence = False) -> List[str]:
        Converts sequences of token indices back to texts.
    """
    def __init__(
        self,
        logger: logging.Logger,
        start_token: str = '<START>',
        end_token: str = '<END>',
        oov_token: str = '',
        max_tokens: int = 150,
        reverse_input_sequence: bool = False
    ) -> None:
        self._logger = logger
        self._start_token = start_token
        self._end_token = end_token
        self._oov_token = oov_token
        self._max_tokens = max_tokens
        self._reverse_input_sequence = reverse_input_sequence

        # Initialize TextVectorization layer
        self.text_vectorization = TextVectorization(
            standardize=None,
            split='whitespace',
            output_mode='int',
            output_sequence_length=None,
            max_tokens=max_tokens,
            pad_to_max_tokens=False,
        )

        self._token_counts = None

    @classmethod
    def from_json(
        cls,
        tokenizer_path: str,
        logger: logging.Logger,
        reverse_input_sequence: bool = False,
        start_token: str = '<START>',
        end_token: str = '<END>',
        oov_token: str = '',
    ) -> 'SmilesTokeniser':
        """
        Loads the tokenizer's vocabulary from a JSON file and initializes a SmilesTokenizer instance.

        Parameters
        ----------
        tokenizer_path : str
            Path to the tokenizer vocabulary JSON file.
        reverse_input_sequence : bool, optional
            Whether to reverse the input sequence during tokenization (default is False).
        start_token : str, optional
            Token representing the start of a sequence (default is '<START>').
        end_token : str, optional
            Token representing the end of a sequence (default is '<END>').
        oov_token : str, optional
            Token representing out-of-vocabulary tokens (default is '').

        Returns
        -------
        SmilesTokeniser
            An instance of SmilesTokenizer with the loaded vocabulary.
        """
        with open(tokenizer_path, 'r') as f:
            word_index = json.load(f)

        # Initialize a new SmilesTokenizer instance
        smiles_tokenizer = cls(
            logger=logger,
            start_token=start_token,
            end_token=end_token,
            oov_token=oov_token,
            max_tokens=len(word_index) + 1,  # +1 to account for padding or OOV
            reverse_input_sequence=reverse_input_sequence
        )
        # Manually set the vocabulary
        smiles_tokenizer.text_vectorization.set_vocabulary(list(word_index.keys()))

        return smiles_tokenizer

    def tokenise_list(
        self,
        smiles_list: List[str],
        is_input_sequence = False
    ) -> List[str]:
        """
        Tokenizes a list of SMILES strings.

        Parameters
        ----------
        smiles_list : List[str]
            A list of SMILES strings.
        is_input_sequence : bool, optional
            Boolean declaring whether SMILES list is sequence input or not.

        Returns
        -------
        List[List[str]]
            A list of token lists.
        """
        return [self._tokenise(smiles, is_input_sequence) for smiles in smiles_list]

    def adapt(self, tokenized_smiles_list: List[str]) -> None:
        """
        Adapts the TextVectorization layer to the preprocessed SMILES list.

        Parameters
        ----------
        tokenized_smiles_list : List[str]
            A list of preprocessed SMILES strings.
        """
        # Convert list to TensorFlow dataset
        self.text_vectorization.adapt(tf.constant(tokenized_smiles_list))

    def texts_to_sequences(self, texts: List[str]) -> tf.Tensor:
        """
        Converts tokenized texts to sequences of token indices.

        Parameters
        ----------
        texts : List[str]
            List of tokenized texts.

        Returns
        -------
        tf.Tensor
            A tensor of token indices.
        """
        return self.text_vectorization(tf.constant(texts))

    def sequences_to_texts(
        self,
        sequences: Union[tf.Tensor, np.ndarray, List[List[int]]],
        is_input_sequence = False
    ) -> List[str]:
        """
        Converts sequences of token indices back to texts.

        Parameters
        ----------
        sequences : Union[tf.Tensor, np.ndarray, List[List[int]]]
            A tensor, NumPy array or list of token indices.
        is_input_sequence : bool, optional
            Boolean declaring whether SMILES list is sequence input or not.

        Returns
        -------
        List[str]
            A list of SMILES strings.
        """
        # Create a reverse mapping from indices to tokens
        vocab = self.text_vectorization.get_vocabulary()
        inverse_vocab = {idx: word for idx, word in enumerate(vocab)}
        texts = []

        if isinstance(sequences, tf.Tensor):
            sequences = sequences.numpy().tolist()
        elif isinstance(sequences, np.ndarray):
            sequences = sequences.tolist()
        elif isinstance(sequences, list):
            pass
        else:
            raise TypeError("Input must be a tf.Tensor, np.ndarray, or list of sequences")

        for sequence in sequences:
            tokens = [inverse_vocab.get(idx, self.oov_token) for idx in sequence if idx != 0]
            # Remove start and end tokens
            if tokens and tokens[0] == self.start_token:
                tokens = tokens[1:]
            if tokens and tokens[-1] == self.end_token:
                tokens = tokens[:-1]
            if self._reverse_input_sequence and is_input_sequence:
                tokens = tokens[::-1]
            texts.append(' '.join(tokens))
        return texts

    def calculate_token_frequencies(self, tokenized_smiles_list) -> Dict:
        token_counts = Counter()
        for line in tokenized_smiles_list:
            tokens = line.split()
            token_counts.update(tokens)

        total_count = sum(token_counts.values())
        self._logger.info(f"Total tokens in dataset: {total_count}")
        self._logger.info(f"Unique tokens: {len(token_counts)}\n")

        # Sort by descending frequency
        for token, count in token_counts.most_common():
            percentage = count / total_count * 100
            self._logger.info(f"Token: {token:10s} | Count: {count:7d} | {percentage:6.2f}%")

        self._token_counts = dict(token_counts)

    def build_token_weight_map(
        self,
        token_counts: Dict,
        alpha: float = 0.1,
        min_count: int = 10
    ):
        """
        Build a 1D float32 tensor of shape [vocab_size],
        where index i corresponds to the weight for token i.

        Example weighting: weight = alpha / (frequency + 1)
        so that rare tokens get higher weights.
        """
        vocab_size = len(self.word_index)
        self._logger.info(f"Building token-to-weight map for {vocab_size} token classes")

        weights_array = np.zeros((vocab_size,), dtype=np.float32)

        # Reverse mapping: idx->token
        idx_to_token = {idx: tok for tok, idx in self.word_index.items()}

        for idx in range(vocab_size):
            token_str = idx_to_token[idx]
            # Default to 1 if the token isn't found in `token_counts` (i.e., a small nonzero freq to avoid dividing
            # by zero)
            freq = token_counts.get(token_str, 1)
            freq = max(freq, min_count)

            # Square root weighting
            ## weights_array[idx] = alpha / math.sqrt(freq)
            ## self._logger.info(f"Token {token_str} weight calculated using square root weighting")

            # Log Weighting
            ## weights_array[idx] = alpha / math.log(float(freq) + math.e)
            ## self._logger.info(f"Token {token_str} weight calculated using log weighting")

            # 'Linear blend' of weights
            weights_array[idx] = 1 + alpha * (1/math.sqrt(freq))
            self._logger.info(f"Token {token_str} weight calculated using 'linear blend' of weights")

        # Normalise weights by dividing all weights by the overall average so that average weight is 1.0
        mean_weight = weights_array.mean()
        weights_array /= mean_weight

        self._logger.info(f"Token-to-weight map successfully built")

        return tf.constant(weights_array, dtype=tf.float32)

    def _tokenise(self, smiles: str, is_input_sequence: bool) -> str:
        """
        Tokenises a single SMILES string into individual characters.

        Parameters
        ----------
        smiles : str
            A SMILES string.
        is_input_sequence : bool, optional
            Is input SMILES string boolean.

        Returns
        -------
        List[str]
            A list of character tokens with start and end tokens.
        """
        basic_smiles_tokenizer = BasicSmilesTokenizer()
        tokenized_smiles = basic_smiles_tokenizer.tokenize(smiles)

        # Add start and end tokens
        tokens = [self.start_token] + tokenized_smiles + [self.end_token]

        # Reverse the SMILES tokens if required (excluding special tokens)
        if self._reverse_input_sequence and is_input_sequence:
            # Reverse only the SMILES tokens, keep start and end tokens in place
            tokens = [self.start_token] + tokenized_smiles[::-1] + [self.end_token]

        # Join tokens back into a string separated by spaces (required for TextVectorization)
        return ' '.join(tokens)

    @property
    def max_tokens(self) -> int:
        """
        Returns the tokeniser vocabulary maximum number of tokens.

        Returns
        -------
        max_tokens: int
            The start token.
        """

        return self._max_tokens

    @property
    def start_token(self) -> str:
        """
        Returns the start token used in tokenization.

        Returns
        -------
        start_token: str
            The start token.
        """
        return self._start_token

    @property
    def end_token(self) -> str:
        """
        Returns the end token used in tokenization.

        Returns
        -------
        end_token: str
            The end token.
        """
        return self._end_token

    @property
    def oov_token(self) -> str:
        """
        Returns the out-of-vocabulary (OOV) token used in tokenization.

        Returns
        -------
        oov_token: str
            The OOV token.
        """
        return self._oov_token

    @property
    def word_index(self) -> Dict[str, int]:
        """
        Gets the word index mapping.

        Returns
        -------
        word_index: dict
            Word to index mapping.
        """
        vocab = self.text_vectorization.get_vocabulary()
        return {word: idx for idx, word in enumerate(vocab)}

    @property
    def vocab_size(self) -> int:
        """
        Gets the size of the vocabulary.

        Returns
        -------
        vocab_size: int
            Vocabulary size.
        """
        return len(self.text_vectorization.get_vocabulary())

    @property
    def token_counts(self) -> Dict:
        """
        Gets the counts for each token in the vocabulary.

        Returns
        -------
        token_counts: Dict
            Token counts.
        """
        return self._token_counts
