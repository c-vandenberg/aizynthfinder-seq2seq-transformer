# AiZynthFinder Seq2Seq and Transformer Expansion Policies

# Abstract

Retrosynthetic analysis and reaction prediction are fundamental for efficient chemical synthesis and drug discovery. Consequently, computer-assisted synthesis planning (CASP) tools have been at the forefront of research and development, striving to computationally identify the optimal sequence of chemical reaction steps that transform commercially viable starting materials into desired chemical compounds. <sup>**1**</sup> 

A leading CASP tool, [AiZynthFinder](https://github.com/MolecularAI/aizynthfinder), achieves retrosynthesis prediction by generating a retrosynthetic search tree using a template-based, feedforward neural network (FNN) model known as the expansion policy to give a ranked list of reaction templates. This process is followed by another neural network, the filter policy, which removes unfeasible reactions. Once the retrosynthesis search tree is constructed, a Monte Carlo Tree Search (MCTS) algorithm traverses the search tree to identify the best synthetic routes.

To enhance AiZynthFinder, this study integrates SMILES-based and attention-based encoder-decoder (Seq2Seq) and transformer models into its expansion policy. By leveraging these advanced neural network architectures with SMILES-based chemical representations, we aim to overcome the inherent limitations of template-based retrosynthetic methods. The integration seeks to broaden accurate predictions beyond the rule-based knowledge base, and ensure that predictions consider the entire molecular environment and account for stereochemistry.

This study is ongoing, involving continuous model optimisations and research. The results and discussion for the latest Seq2Seq model are available [here](https://github.com/c-vandenberg/aizynthfinder-project/blob/master/expansion-policies/seq2seq-expansion-policy/src/models/README.md#55-results-and-discussion). Development of the transformer model is currently in progress.

# Contents

<details>
  <summary><b>1. Retrosynthesis with AiZynthFinder - Overview</b></summary>
  
  &nbsp; &nbsp; &nbsp; &nbsp; 1.1 [Basics of Retrosynthesis](https://github.com/c-vandenberg/aizynthfinder-project?tab=readme-ov-file#11-basics-of-retrosynthesis)<br>
  &nbsp; &nbsp; &nbsp; &nbsp; 1.2 [Retrosynthetic Search Tree](https://github.com/c-vandenberg/aizynthfinder-project?tab=readme-ov-file#12-retrosynthetic-search-tree)<br>
  &nbsp; &nbsp; &nbsp; &nbsp; 1.3 [AiZynthFinder Template-Based Retrosynthesis Model (Define Disconnection Rules)](https://github.com/c-vandenberg/aizynthfinder-project?tab=readme-ov-file#13-aizynthfinder-template-based-retrosynthesis-model-define-disconnection-rules)<br>
  &nbsp; &nbsp; &nbsp; &nbsp; 1.4 [Monte Carlo Tree Search: Finding the Best Routes (Traverse the Retrosynthesis Search Tree Efficiently)](https://github.com/c-vandenberg/aizynthfinder-project?tab=readme-ov-file#14-monte-carlo-tree-search-finding-the-best-routes-traverse-the-retrosynthesis-search-tree-efficiently)<br>
  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 1.4.1 [Heuristic Search Algorithms](https://github.com/c-vandenberg/aizynthfinder-project?tab=readme-ov-file#141-heuristic-search-algorithms)<br>
  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 1.4.2 [Monte Carlo Tree Search in AiZynthFinder](https://github.com/c-vandenberg/aizynthfinder-project?tab=readme-ov-file#142-monte-carlo-tree-search-in-aizynthfinder)<br>
  &nbsp; &nbsp; &nbsp; &nbsp; 1.5 [AiZynthFinder Post-Processing Tools - Route Scoring](https://github.com/c-vandenberg/aizynthfinder-project?tab=readme-ov-file#15-aizynthfinder-post-processing-tools---route-scoring)<br>
  &nbsp; &nbsp; &nbsp; &nbsp; 1.6 [Route Clustering](https://github.com/c-vandenberg/aizynthfinder-project?tab=readme-ov-file#16-route-clustering)<br>
  &nbsp; &nbsp; &nbsp; &nbsp; 1.7 [References](https://github.com/c-vandenberg/aizynthfinder-project?tab=readme-ov-file#17-references)<br>
</details>
<details>
  <summary><b>2. AiZynthFinder's Expansion Policy Neural Network</b></summary>
  
  &nbsp; &nbsp; &nbsp; &nbsp; 2.1 [What is AiZynthFinder's Expansion Policy Neural Network?](https://github.com/c-vandenberg/aizynthfinder-project/tree/master/expansion-policies#21-what-is-aizynthfinders-expansion-policy-neural-network)<br>
  &nbsp; &nbsp; &nbsp; &nbsp; 2.2 [Neural Networks Overview](https://github.com/c-vandenberg/aizynthfinder-project/tree/master/expansion-policies#22-neural-networks-overview)<br>
  &nbsp; &nbsp; &nbsp; &nbsp; 2.3 [Feedforward Neural Networks (FNNs)](https://github.com/c-vandenberg/aizynthfinder-project/tree/master/expansion-policies#23-feedforward-neural-networks-fnns)<br>
  &nbsp; &nbsp; &nbsp; &nbsp; 2.4 [Recurrent Neural Networks (RNNs)](https://github.com/c-vandenberg/aizynthfinder-project/blob/master/expansion-policies/README.md#24-recurrent-neural-networks-rnns)<br>
  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 2.4.1 [Recurrent Neural Network Architecture](https://github.com/c-vandenberg/aizynthfinder-project/tree/master/expansion-policies#241-recurrent-neural-network-architecture)<br>
  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 2.4.2 [Backpropagation vs Backpropagation Through Time](https://github.com/c-vandenberg/aizynthfinder-project/tree/master/expansion-policies#242-backpropagation-vs-backpropagation-through-time)<br>
  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 2.4.3 [Recurrent Neural Network Training](https://github.com/c-vandenberg/aizynthfinder-project/tree/master/expansion-policies#242-recurrent-neural-network-training)<br>
  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 2.4.4 [Types of Recurrent Neural Networks](https://github.com/c-vandenberg/aizynthfinder-project/blob/master/expansion-policies/README.md#244-types-of-recurrent-neural-networks)<br>
  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; i. [Standard RNNs](https://github.com/c-vandenberg/aizynthfinder-project/blob/master/expansion-policies/README.md#i-standard-rnns)<br>
  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; ii. [Bidirectional Recurrent Neural Networks (BRRNs)](https://github.com/c-vandenberg/aizynthfinder-project/blob/master/expansion-policies/README.md#ii-bidirectional-recurrent-neural-networks-brrns)<br>
  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; iii. [Long Short-Term Memory (LSTM)](https://github.com/c-vandenberg/aizynthfinder-project/blob/master/expansion-policies/README.md#iii-long-short-term-memory-lstm)<br>
  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; iv. [Gated Recurrent Units (GNUs)](https://github.com/c-vandenberg/aizynthfinder-project/blob/master/expansion-policies/README.md#iv-gated-recurrent-units-gnus)<br>
  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; v. [Encoder-Decoder RNN](https://github.com/c-vandenberg/aizynthfinder-project/blob/master/expansion-policies/README.md#v-encoder-decoder-rnn)<br>
  &nbsp; &nbsp; &nbsp; &nbsp; 2.5 [References](https://github.com/c-vandenberg/aizynthfinder-project/tree/master/expansion-policies#24-references)<br>
</details>
<details>
  <summary><b>3. Attention-Based Encoder-Decoder (Seq2Seq) Expansion Policy</b></summary>
  
  &nbsp; &nbsp; &nbsp; &nbsp; 3.1 [Limitations of Template-Based Retrosynthetic Methods](https://github.com/c-vandenberg/aizynthfinder-project/tree/master/expansion-policies/seq2seq-expansion-policy#31-limitations-of-template-based-retrosynthetic-methods)<br>
  &nbsp; &nbsp; &nbsp; &nbsp; 3.2 [Alternative SMILES-Based Retrosynthetic Methods](https://github.com/c-vandenberg/aizynthfinder-project/blob/master/expansion-policies/seq2seq-expansion-policy/README.md#32-alternative-smiles-based-retrosynthetic-methods)<br>
  &nbsp; &nbsp; &nbsp; &nbsp; 3.3 [Sequence-to-Sequence Model](https://github.com/c-vandenberg/aizynthfinder-project/tree/master/expansion-policies/seq2seq-expansion-policy#33-sequence-to-sequence-model)<br>
  &nbsp; &nbsp; &nbsp; &nbsp; 3.4 [Architecture of Sequence-to-Sequence Models](https://github.com/c-vandenberg/aizynthfinder-project/tree/master/expansion-policies/seq2seq-expansion-policy#34-architecture-of-sequence-to-sequence-models)<br>
  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 3.4.1 [Encoder](https://github.com/c-vandenberg/aizynthfinder-project/tree/master/expansion-policies/seq2seq-expansion-policy#341-encoder)<br>
  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 3.4.2 [Decoder](https://github.com/c-vandenberg/aizynthfinder-project/tree/master/expansion-policies/seq2seq-expansion-policy#342-decoder)<br>
  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 3.4.3 [Attention Mechanism](https://github.com/c-vandenberg/aizynthfinder-project/tree/master/expansion-policies/seq2seq-expansion-policy#343-attention-mechanism)<br>
  &nbsp; &nbsp; &nbsp; &nbsp; 3.5 [References](https://github.com/c-vandenberg/aizynthfinder-project/tree/master/expansion-policies/seq2seq-expansion-policy#35-references)<br>
</details>
<details>
  <summary><b>4. Retrosynthesis Sequence-to-Sequence Model Literature Review</b></summary>

  &nbsp; &nbsp; &nbsp; &nbsp; 4.1 [*Britz et al.* Analysis of Neural Machine Translation Architecture Hyperparameters](https://github.com/c-vandenberg/aizynthfinder-project/blob/master/expansion-policies/seq2seq-expansion-policy/src/README.md#41-britz-et-al-analysis-of-neural-machine-translation-architecture-hyperparameters)<br>
  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 4.1.1 [Embedding Dimensionality](https://github.com/c-vandenberg/aizynthfinder-project/blob/master/expansion-policies/seq2seq-expansion-policy/src/README.md#411-embedding-dimensionality)<br>
  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 4.1.2 [Encoder and Decoder Recurrent Neural Network (RNN) Cell Variant](https://github.com/c-vandenberg/aizynthfinder-project/blob/master/expansion-policies/seq2seq-expansion-policy/src/README.md#412-encoder-and-decoder-recurrent-neural-network-rnn-cell-variant)<br>
  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 4.1.3 [Encoder and Decoder Depth](https://github.com/c-vandenberg/aizynthfinder-project/blob/master/expansion-policies/seq2seq-expansion-policy/src/README.md#413-encoder-and-decoder-depth)<br>
  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 4.1.4 [Unidirectional vs. Bidirectional Encoder](https://github.com/c-vandenberg/aizynthfinder-project/blob/master/expansion-policies/seq2seq-expansion-policy/src/README.md#414-unidirectional-vs-bidirectional-encoder)<br>
  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 4.1.5 [Attention Mechanism](https://github.com/c-vandenberg/aizynthfinder-project/blob/master/expansion-policies/seq2seq-expansion-policy/src/README.md#415-attention-mechanism)<br>
  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 4.1.6 [Beam Search Strategies](https://github.com/c-vandenberg/aizynthfinder-project/blob/master/expansion-policies/seq2seq-expansion-policy/src/README.md#416-beam-search-strategies)<br>
  &nbsp; &nbsp; &nbsp; &nbsp; 4.2 [*Liu et al.* Sequence-to-Sequence Model](https://github.com/c-vandenberg/aizynthfinder-project/blob/master/expansion-policies/seq2seq-expansion-policy/src/README.md#42-liu-et-al-sequence-to-sequence-model)<br>
  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 4.1.1 [Data Preparation](https://github.com/c-vandenberg/aizynthfinder-project/blob/master/expansion-policies/seq2seq-expansion-policy/src/README.md#411-data-preparation)<br>
  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; i. [Training](https://github.com/c-vandenberg/aizynthfinder-project/blob/master/expansion-policies/seq2seq-expansion-policy/src/README.md#i-training)<br>
  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; ii. [Testing](https://github.com/c-vandenberg/aizynthfinder-project/blob/master/expansion-policies/seq2seq-expansion-policy/src/README.md#ii-testing)<br>
  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 4.1.2 [Model Architecture](https://github.com/c-vandenberg/aizynthfinder-project/blob/master/expansion-policies/seq2seq-expansion-policy/src/README.md#412-model-architecture)<br>
  &nbsp; &nbsp; &nbsp; &nbsp; 4.3 [References](https://github.com/c-vandenberg/aizynthfinder-project/blob/master/expansion-policies/seq2seq-expansion-policy/src/README.md#43-references)<br>
</details>
<details>
  <summary><b>5. Project Retrosynthesis Sequence-to-Sequence Model</b></summary>
  
  &nbsp; &nbsp; &nbsp; &nbsp; 5.1 [Data Preparation](https://github.com/c-vandenberg/aizynthfinder-project/blob/master/expansion-policies/seq2seq-expansion-policy/src/models/README.md#51-data-preparation)<br>
  &nbsp; &nbsp; &nbsp; &nbsp; 5.2 [Model Optimisation](https://github.com/c-vandenberg/aizynthfinder-project/blob/master/expansion-policies/seq2seq-expansion-policy/src/models/README.md#52-model-optimisation)<br>
  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 5.2.1 [Deterministic Training Environment](https://github.com/c-vandenberg/aizynthfinder-project/blob/master/expansion-policies/seq2seq-expansion-policy/src/models/README.md#521-deterministic-training-environment)<br>
  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 5.2.2 [Data Tokenization and Preprocessing Optimisation](https://github.com/c-vandenberg/aizynthfinder-project/blob/master/expansion-policies/seq2seq-expansion-policy/src/models/README.md#522-data-tokenization-and-preprocessing-optimisation)<br>
  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; i. [DeepChem Tokenizer](https://github.com/c-vandenberg/aizynthfinder-project/blob/master/expansion-policies/seq2seq-expansion-policy/src/models/README.md#i-deepchem-tokenizer)<br>
  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; ii. [TensorFlow TextVectorisation](https://github.com/c-vandenberg/aizynthfinder-project/blob/master/expansion-policies/seq2seq-expansion-policy/src/models/README.md#ii-tensorflow-textvectorisation)<br>
  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 5.2.3 [Loss Function Optimisation](https://github.com/c-vandenberg/aizynthfinder-project/blob/master/expansion-policies/seq2seq-expansion-policy/src/models/README.md#523-loss-function-optimisation)<br>
  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; i. [Categorical Cross-Entropy vs Sparse Categorical Cross-Entropy](https://github.com/c-vandenberg/aizynthfinder-project/blob/master/expansion-policies/seq2seq-expansion-policy/src/models/README.md#i-categorical-cross-entropy-vs-sparse-categorical-cross-entropy)<br>
  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; ii. [Optimiser - Adam](https://github.com/c-vandenberg/aizynthfinder-project/blob/master/expansion-policies/seq2seq-expansion-policy/src/models/README.md#ii-optimiser---adam)<br>
  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; iii. [Weight Decay (L2 Regularisation)](https://github.com/c-vandenberg/aizynthfinder-project/blob/master/expansion-policies/seq2seq-expansion-policy/src/models/README.md#iii-weight-decay-l2-regularisation)<br>
  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 5.2.4 [Callbacks Optimisation](https://github.com/c-vandenberg/aizynthfinder-project/blob/master/expansion-policies/seq2seq-expansion-policy/src/models/README.md#524-callbacks-optimisation)<br>
  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; i. [EarlyStopping](https://github.com/c-vandenberg/aizynthfinder-project/blob/master/expansion-policies/seq2seq-expansion-policy/src/models/README.md#i-earlystopping)<br>
  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; ii. [Dynamic Learning Rate (ReduceLROnPlateau)](https://github.com/c-vandenberg/aizynthfinder-project/blob/master/expansion-policies/seq2seq-expansion-policy/src/models/README.md#ii-dynamic-learning-rate-reducelronplateau)<br>
  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; iii. [Checkpoints (ModelCheckpoint)](https://github.com/c-vandenberg/aizynthfinder-project/blob/master/expansion-policies/seq2seq-expansion-policy/src/models/README.md#iii-checkpoints-modelcheckpoint)<br>
  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; iv. [Visualisation in TensorBoard (TensorBoard)](https://github.com/c-vandenberg/aizynthfinder-project/blob/master/expansion-policies/seq2seq-expansion-policy/src/models/README.md#iv-visualisation-in-tensorboard-tensorboard)<br>
  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; v. [Validation Metrics (ValidationMetricsCallback)](https://github.com/c-vandenberg/aizynthfinder-project/blob/master/expansion-policies/seq2seq-expansion-policy/src/models/README.md#v-validation-metrics-validationmetricscallback)<br>
  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 5.2.5 [Metrics Optimisation](https://github.com/c-vandenberg/aizynthfinder-project/blob/master/expansion-policies/seq2seq-expansion-policy/src/models/README.md#525-metrics-optimisation)<br>
  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; i. [Perplexity](https://github.com/c-vandenberg/aizynthfinder-project/blob/master/expansion-policies/seq2seq-expansion-policy/src/models/README.md#i-perplexity)<br>
  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; ii. [BLEU Score](https://github.com/c-vandenberg/aizynthfinder-project/blob/master/expansion-policies/seq2seq-expansion-policy/src/models/README.md#ii-bleu-score)<br>
  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; iii. [SMILES String Metrics (Exact Match, Chemical Validity, Tanimoto Similarity, Levenshtein Distance)](https://github.com/c-vandenberg/aizynthfinder-project/blob/master/expansion-policies/seq2seq-expansion-policy/src/models/README.md#iii-smiles-string-metrics-exact-match-chemical-validity-tanimoto-similarity-levenshtein-distance)<br>
  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 5.2.6 [Encoder Optimisation](https://github.com/c-vandenberg/aizynthfinder-project/blob/master/expansion-policies/seq2seq-expansion-policy/src/models/README.md#526-encoder-optimisation)<br>
  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; i. [Residual Connections](https://github.com/c-vandenberg/aizynthfinder-project/blob/master/expansion-policies/seq2seq-expansion-policy/src/models/README.md#i-residual-connections)<br>
  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; ii. [Layer Normalisation](https://github.com/c-vandenberg/aizynthfinder-project/blob/master/expansion-policies/seq2seq-expansion-policy/src/models/README.md#ii-layer-normalisation)<br>
  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 5.2.7 [Decoder Optimisation](https://github.com/c-vandenberg/aizynthfinder-project/blob/master/expansion-policies/seq2seq-expansion-policy/src/models/README.md#527-decoder-optimisation)<br>
  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; i. [Residual Connections and Layer Normalisation](https://github.com/c-vandenberg/aizynthfinder-project/blob/master/expansion-policies/seq2seq-expansion-policy/src/models/README.md#i-residual-connections-and-layer-normalisation)<br>
  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 5.2.8 [Attention Mechanism Optimisation](https://github.com/c-vandenberg/aizynthfinder-project/blob/master/expansion-policies/seq2seq-expansion-policy/src/models/README.md#538-attention-mechanism-optimisation)<br>
  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; i. [Bahdanau Attention Mechanism](https://github.com/c-vandenberg/aizynthfinder-project/blob/master/expansion-policies/seq2seq-expansion-policy/src/models/README.md#i-bahdanau-attention-mechanism)<br>
  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; ii. [Residual Connections and Layer Normalisation](https://github.com/c-vandenberg/aizynthfinder-project/blob/master/expansion-policies/seq2seq-expansion-policy/src/models/README.md#ii-residual-connections-and-layer-normalisation)<br>
  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 5.2.9 [Inference Optimisation](https://github.com/c-vandenberg/aizynthfinder-project/blob/master/expansion-policies/seq2seq-expansion-policy/src/models/README.md#529-inference-optimisation)<br>
  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; i. [Greedy Decoding vs Beam Search](https://github.com/c-vandenberg/aizynthfinder-project/blob/master/expansion-policies/seq2seq-expansion-policy/src/models/README.md#i-greedy-decoding-vs-beam-search)<br>
  &nbsp; &nbsp; &nbsp; &nbsp; 5.3 [Model Architecture](https://github.com/c-vandenberg/aizynthfinder-project/blob/master/expansion-policies/seq2seq-expansion-policy/src/models/README.md#53-model-architecture)<br>
  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 5.3.1 [Optimised Encoder Architecture](https://github.com/c-vandenberg/aizynthfinder-project/blob/master/expansion-policies/seq2seq-expansion-policy/src/models/README.md#531-optimised-encoder-architecture)<br>
  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 5.3.2 [Optimised Decoder Architecture](https://github.com/c-vandenberg/aizynthfinder-project/blob/master/expansion-policies/seq2seq-expansion-policy/src/models/README.md#532-optimised-decoder-architecture)<br>
  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 5.3.3 [Optimised Attention Mechanism (Bahdanau Attention) Architecture](https://github.com/c-vandenberg/aizynthfinder-project/blob/master/expansion-policies/seq2seq-expansion-policy/src/models/README.md#533-optimised-attention-mechanism-bahdanau-attention-architecture)<br>
  &nbsp; &nbsp; &nbsp; &nbsp; 5.4 [Model Documentation](https://github.com/c-vandenberg/aizynthfinder-project/blob/master/expansion-policies/seq2seq-expansion-policy/src/models/README.md#54-model-documentation)<br>
  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 5.4.1 [Model Training Pipeline](https://github.com/c-vandenberg/aizynthfinder-project/blob/master/expansion-policies/seq2seq-expansion-policy/src/models/README.md#541-model-training-pipeline)<br>
  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 5.4.2 [Model Data Flow - ONNX Graph](https://github.com/c-vandenberg/aizynthfinder-project/blob/master/expansion-policies/seq2seq-expansion-policy/src/models/README.md#542-model-data-flow---onnx-graph)<br>
  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; i. [Flow of Data Through Encoder](https://github.com/c-vandenberg/aizynthfinder-project/blob/master/expansion-policies/seq2seq-expansion-policy/src/models/README.md#i-flow-of-data-through-encoder)<br>
  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; ii. [Flow of Data Through Decoder](https://github.com/c-vandenberg/aizynthfinder-project/blob/master/expansion-policies/seq2seq-expansion-policy/src/models/README.md#ii-flow-of-data-through-decoder)<br>
  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; iii. [Flow of Data Through Attention Mechanism](https://github.com/c-vandenberg/aizynthfinder-project/blob/master/expansion-policies/seq2seq-expansion-policy/src/models/README.md#iii-flow-of-data-through-attention-mechanism)<br>
  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 5.4.3 [Model Debugging](https://github.com/c-vandenberg/aizynthfinder-project/blob/master/expansion-policies/seq2seq-expansion-policy/src/models/README.md#543-model-debugging)<br>
  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; i. [Data Tokenization and Preprocessing Debugging](https://github.com/c-vandenberg/aizynthfinder-project/blob/master/expansion-policies/seq2seq-expansion-policy/src/models/README.md#i-data-tokenization-and-preprocessing-debugging)<br>
  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; ii. [General TensorFlow Debugging](https://github.com/c-vandenberg/aizynthfinder-project/blob/master/expansion-policies/seq2seq-expansion-policy/src/models/README.md#ii-general-tensorflow-debugging)<br>
  &nbsp; &nbsp; &nbsp; &nbsp; 5.5 [Results and Discussion](https://github.com/c-vandenberg/aizynthfinder-project/blob/master/expansion-policies/seq2seq-expansion-policy/src/models/README.md#55-results-and-discussion)<br>
  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 5.5.1 [Evaluation of Current Optimal Model Architecture](https://github.com/c-vandenberg/aizynthfinder-project/blob/master/expansion-policies/seq2seq-expansion-policy/src/models/README.md#551-evaluation-of-current-optimal-model-architecture)<br>
  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; i. [Analysis of Performance Metrics](https://github.com/c-vandenberg/aizynthfinder-project/blob/master/expansion-policies/seq2seq-expansion-policy/src/models/README.md#i-analysis-of-performance-metrics)<br>
  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; ii. [Analysis of Sample Predictions](https://github.com/c-vandenberg/aizynthfinder-project/blob/master/expansion-policies/seq2seq-expansion-policy/src/models/README.md#ii-analysis-of-sample-predictions)<br>
  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 5.5.2 [Integrating Seqeuence-to-Sequence Model into AiZynthFinder](https://github.com/c-vandenberg/aizynthfinder-project/blob/master/expansion-policies/seq2seq-expansion-policy/src/models/README.md#552-integrating-seqeuence-to-sequence-model-into-aizynthfinder)<br>
  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; i. [Simple Drug Retrosynthesis - Aspirin](https://github.com/c-vandenberg/aizynthfinder-project/blob/master/expansion-policies/seq2seq-expansion-policy/src/models/README.md#i-simple-drug-retrosynthesis---aspirin)<br>
  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; ii. [Complex Chiral Drug Retrosynthesis - Rivaroxaban](https://github.com/c-vandenberg/aizynthfinder-project/blob/master/expansion-policies/seq2seq-expansion-policy/src/models/README.md#ii-complex-chiral-drug-retrosynthesis---rivaroxaban)<br>
  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; iii. [AiZynthFinder Expansion Policy Performance Analysis](https://github.com/c-vandenberg/aizynthfinder-project/blob/master/expansion-policies/seq2seq-expansion-policy/src/models/README.md#iii-aizynthfinder-expansion-policy-performance-analysis)<br>
  &nbsp; &nbsp; &nbsp; &nbsp; 5.6 [Future Model Optimisations](https://github.com/c-vandenberg/aizynthfinder-project/blob/master/expansion-policies/seq2seq-expansion-policy/src/models/README.md#56-future-model-optimisations)<br>
  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 5.6.1 [Increased Training Dataset Size and Diversity](https://github.com/c-vandenberg/aizynthfinder-project/blob/master/expansion-policies/seq2seq-expansion-policy/src/models/README.md#561-increased-training-dataset-size-and-diversity)<br>
  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 5.6.2 [Layer-wise Learning Rate Decay](https://github.com/c-vandenberg/aizynthfinder-project/blob/master/expansion-policies/seq2seq-expansion-policy/src/models/README.md#562-layer-wise-learning-rate-decay)<br>
  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 5.6.3 [Scheduled Sampling](https://github.com/c-vandenberg/aizynthfinder-project/blob/master/expansion-policies/seq2seq-expansion-policy/src/models/README.md#563-scheduled-sampling)<br>
  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 5.6.4 [High Throughput Testing of Model Expansion Policy Performance](https://github.com/c-vandenberg/aizynthfinder-project/blob/master/expansion-policies/seq2seq-expansion-policy/src/models/README.md#564-high-throughput-testing-of-model-expansion-policy-performance)<br>
  &nbsp; &nbsp; &nbsp; &nbsp; 5.7 [References](https://github.com/c-vandenberg/aizynthfinder-project/blob/master/expansion-policies/seq2seq-expansion-policy/src/models/README.md#57-references)<br>
</details>
<details>
  <summary><b>6. Transformer Expansion Policy</b></summary>
</details>
<details>
  <summary><b>7. Retrosynthesis Transformer Model Literature Review</b></summary>
</details>
<details>
  <summary><b>8. Project Retrosynthesis Transformer Model</b></summary>
</details>

# 1. Retrosynthesis with AiZynthFinder - Overview

**AiZynthFinder** is a **computer-aided synthesis planning (CASP)** tool developed by **AstraZeneca's MolecularAI department**. Specifically, it is a **computer-assisted synthesis prediction tool** that seeks to identify the **optimal sequence of chemical reaction steps** capable of transforming a set of **commercially available starting materials** into a **desired chemical compound**. **<sup>1</sup>**  **<sup>2</sup>**

AiZynthFinder leverages recent advancements in **machine learning techniques**, specifically **deep neural networks**, to **predict synthetic pathways via retrosynthetic analysis** with **minimal human intervention**. **<sup>1</sup>**  **<sup>3</sup>**

## 1.1 Basics of Retrosynthesis 

**Retrosynthetic analysis** involves the **deconstruction of a target molecule** into **simpler precursor structures** in order to **probe different synthetic routes** to the target molecule and **compare the different routes** in terms of synthetic viability.

Retrosynthesis involves:
1. **Disconnection**:
   * The **breaking of a chemical bond** to give a **possible starting material**. This can be thought of as the reverse of a synthetic reaction.
<br>
  <div align="center">
    <img src="https://github.com/user-attachments/assets/292e43d1-9b0e-47ea-ae8e-0c1bda3fe67b", alt="retrosynthesis-disconnection"/>
    <p>
      <b>Fig 1</b> Disconnection in retrosynthesis.
    </p>
  </div>
<br>

2. **Synthons**:
   * These are the **fragments produced by the disconnection**.
   * Usually, a single bond disconnection will give a **negatively charged, nucleophilic synthon**, and a **positively charged, electrophilic synthon**.
<br>
  <div align="center">
    <img src="https://github.com/user-attachments/assets/ee9ed362-1d6d-4de9-a1c8-ad6d8084ea9a", alt="retrosynthesis-synthons-ionic"/>
    <p>
      <b>Fig 2</b> Ionic synthons in retrosynthesis.
    </p>
  </div>
<br>

   * However, other times the disconnection will give **neutral fragments**. Classical examples of this are **pericyclic reactions**, such as **Diels-Alder reactions**.
<br>
  <div align="center">
    <img src="https://github.com/user-attachments/assets/0715c148-cbb2-4ba2-b1f3-ece94738276c", alt="retrosynthesis-synthons-neutral"/>
    <p>
      <b>Fig 3</b> Neutral synthons in retrosynthesis.
    </p>
  </div>
<br>

3. **Synthetic Equivalents**:
   * Synthons are not species that exist in reality due to their **reactivity**, and so a synthetic equivalent is a **reagent carrying out the function of a synthon** in the synthesis.
<br>

4. **Functional Group Interconversion (FGI)**:
   * If a **disconnection is not possible** at a given site, **FGI can be used**.
   * An FGI is an operation whereby **one functional group is converted into another** so that a **disconnection becomes possible**.
   * A common FGI is the **oxidation of an alcohol to a carbonyl, or amine to nitro group**
<br>
  <div align="center">
    <img src="https://github.com/user-attachments/assets/321dc552-e38a-422f-885a-104fbe11d56f", alt="functional-group-interconversion"/>
    <p>
      <b>Fig 4</b> Functional group interconversion (FGI) in retrosynthesis.
    </p>
  </div>
<br>
     
5. **Functional Group Addition (FGA)**:
   * Similar to FGI, **FGA** is the **addition of a functional group to another** to make it **suitable for disconnection**.
<br>
  <div align="center">
    <img src="https://github.com/user-attachments/assets/e56e3767-d38e-40a1-a963-0dad681f4078", alt="functional-group-addition"/>
    <p>
      <b>Fig 5</b> Functional group addition (FGA) in retrosynthesis.
    </p>
  </div>
<br>

## 1.2 Retrosynthetic Search Tree

Typically, the retrosynthetic analysis of a target molecule is an **iterative process** whereby the **subsequent fragments are themselves broken down** until we **reach a stop criterion**. This stop criterion is typically when we reach **precursors that are commercially available/in stock**.

<br>
  <div align="center">
    <img src="https://github.com/user-attachments/assets/bb4d5580-028b-4610-b95e-8b2f4fc234ba", alt="retrosynthetic-tree"/>
    <p>
      <b>Fig 6</b> Chemical representation of a retrosynthesis search tree. <b><sup>4</sup></b>
    </p>
  </div>
<br>

This iterative process results in a **retrosynthesis tree** where the **breadth is incredibly large**, but the **depth is quite small/shallow**. In comparison to the search trees for games such as chess and Go (**Fig 6**), the **breadth of a retrosynthesis search tree is incredibly large** because you could **theoretically break any bonds** in the target molecule, and the subsequent fragments. This leads to an **explosion in child nodes** from the **first few subtrees**.

The **depth** of a retrosynthesis search tree is **small/shallow** on the other hand, as it only takes a **few disconnections before viable precursors are found**. This is ideal since we don't want **linear synthetic reactions** with an **excessive number of steps**.

<br>
  <div align="center">
    <img src="https://github.com/user-attachments/assets/6c3efa40-bf2c-4124-bd13-4ad9f5a5d6b0", alt="retrosynthetic-search-tree-breadth-depth"/>
    <p>
      <b>Fig 7</b> Retrosynthesis search tree breadth and depth compared to the search trees in chess and Go. <b><sup>5</sup></b>
    </p>
  </div>
<br>

For **effective retrosynthetic analysis**, a retrosynthesis program must:
1. **Define the disconnection rules clearly and efficiently** in order to **reduce the breadth** of the retrosynthesis search tree.
2. **Traverse the retrosynthesis search tree efficiently** using an **effective search algorithm**.

### 1.3 AiZynthFinder Template-Based Retrosynthesis Model (Define Disconnection Rules)

AiZynthFinder uses a **template-based retrosynthesis model** to **define the disconnection rules**. This approach utilises a **curated database** of **transformation rules** that are **extracted from external reaction databases** that are then **encoded computationally into reaction templates** as **SMIRKS**. 
* **SMIRKS** is a form of **linear notation** used for **molecular reaction representation**. It was developed by **Daylight** and can be thought of as a hybrid between **SMILES** and **SMARTS**.

These reaction templates can then be used as the **disconnection rules** for decomposing the target molecule into simpler, commercially available precursors.

However, before they are used, AiZynthFinder uses a **simple neural network (Expansion policy)** to **predict the probability for each template given a molecule** **<sup>6</sup>** (**Fig 8**).

<br>
  <div align="center">
    <img src="https://github.com/user-attachments/assets/8e60a52c-d9b1-474d-ad72-faf6f8592626", alt="aizynthfinder-template-neural-network"/>
    <p>
      <b>Fig 8</b> Reaction template ranking using Expansion policy neural network. <b><sup>5</sup></b>
    </p>
  </div>
<br>

This **expansion policy neural network template ranking** works as follows:
1. **Encoding of query molecule/target molecule**: The query molecule/target molecule is encoded as an **extended-connectivity fingerprint (ECFP) bit string**, **<sup>7</sup>** specifically an **ECFP4 bit string**.
2. **Expansion policy neural network**: The ECFP4 fingerprints are then **fed into a simple neural network**, called an **expansion policy**. The **output of this neural network** is a **ranked list of templates**.
3. **Keep top-ranked templates and apply to target molecule**: The top-ranked templates are kept (typically the **top 50**), and are **applied to the target molecule**, producing **different sets of precursors**

However, the expansion policy **doesn't know much about chemistry** and **doesn't take all of the reaction environment into consideration**. As a result, it can **rank unfeasible reactions highly**.

Therefore, AiZynthFinder has **another trained neural network** called **filter policy** that is used to **filter and remove unfeasible reactions** (**Fig 9**). **<sup>8</sup>**

<br>
  <div align="center">
    <img src="https://github.com/user-attachments/assets/2ab97b1a-54b8-48cd-bd46-2f32c74d82b0", alt="aizynthfinder-neural-network-reaction-filter"/>
    <p>
      <b>Fig 9</b> <b>a)</b> An example suggested route from the expansion policy without the filter neural network. The single step route would not practical in the wet-laboratory however due to selectivity issues. <b>b)</b> The suggested route when the filter neural network is applied. Although not perfect, it is a much more feasible route. <b><sup>8</sup></b>
    </p>
  </div>
<br>

## 1.4 Monte Carlo Tree Search: Finding the Best Routes (Traverse the Retrosynthesis Search Tree Efficiently)

### 1.4.1 Heuristic Search Algorithms

**Monte Carlo Tree Search (MCTS)** **<sup>9</sup>** is a powerful search algorithm that uses **heuristics** (i.e., **rules of thumb**) for **decision-making processes**, particularly in **complex search spaces**.

  * Like with other **heuristic search algorithms**, the goal of MCTS is to **find a good enough solution within a reasonable amount of time**, rather than **guaranteeing the optimal solution** by **examining all possible outcomes**.
  * Heuristic search algorithms like MCTS are guided by a **heuristic function**, which is a mathematical function used to estimate the **cost, distance or likelihood of reaching the goal from a given state or node**. This function helps **prioritse which paths or options to explore**, based on their likelihood of **leading to an optimal or near-optimal solution**.
  * Heuristic search algorithms aim to **reduce the search space**, making them **more efficient than exhaustive search methods**. By **focusing on promising paths**, they can often **find solutions faster**, especially in complex or large problem spaces. Although these solutions **may not be optimal**, they are **usually good enough for practical purposes**.

### 1.4.2 Monte Carlo Tree Search in AiZynthFinder

In AiZynthFinder, MCTS plays a crucial role in **effectively navigating the vast search space of possible synthetic routes** to find the **best synthesis pathway** for a target molecule.

To recap, the **retrosynthesis tree structure representation** consists of:
* **Nodes**: Each node in the tree represents **a state of the retrosynthesis problem**. In AiZynthFinder, a node corresponds to **a set of one or more intermediate molecules** that can be used to **synthesise the molecule(s)** in the **current node's parent node**.
* **Edges**: The edges between the nodes represent the **application of a specific reaction template (disconnection rule)** to **decompose the molecule set in the parent node** into **simpler precursor molecules in the child node**.

In AiZynthFinder, MCTS uses **iterative/sequential Monte Carlo simulations** **<sup>10</sup>** to explore potential synthetic routes as follows:

**1. Selection**
* Starting at the **root node** (target molecule), the MCTS algorithm selects the **most promising node for expansion** based on a balance of **exploration (trying new reactions)**, and **exploitation (further exploring known good reactions)**
* This is governed by a **Upper Confidence Bound (UCB) score formula** (**Fig 10**) and is how AiZynthFinder **selects and scores routes**.

<br>
  <div align="center">
    <img src="https://github.com/user-attachments/assets/c062ee48-e9cc-468d-90ea-3e5001ad40e9", alt="upper-confidence-bound-score"/>
    <p>
      <b>Fig 10</b> AiZynthFinder Upper Confidence Bound (UCB) score formula for selecting and scoring synthetic routes in a retrosynthesis tree. <b><sup>5</sup></b>
    </p>
  </div>
<br>

**2. Expansion**
* Once a node has been selected, the MCTS algorithm **expands it** by applying a **new reaction template** from the **expansion policy** and **filter policy**, generating **new precursor molecules** (i.e., **new child nodes**).
* At **each expansion step**, the **expansion policy and filter policy** are used to **filter out unfeasible reactions**, ensuring that the search **focuses on viable synthetic routes**.

**3. Rollout/Iteration**
* This process of **selection and expansion** is then **repeated for each resulting precursor molecule** until the **stop criterion is met** and we reach a **terminal node**.
* The stop criterion is usually either when the search reaches **commercially available precursors**, or when it reaches a **pre-defined tree depth/number of disconnections**.

**4. Update/Backpropagation**
* Once the **terminal node is reached**, the **Monte Carlo simulation** is complete and a **complete synthetic route is generated**. This completed simulation/synthetic route is known as a **playout**.
* The **score of the terminal node** (and hence the **score of the playout/synthetic route**) is then **propagated up through the tree**.
* This score of the terminal node is the **accumulated reward (Q)** in the **UCB Score formula** (**Fig 10**), which is a function of the **tree depth** at the terminal node (i.e., **how many synthesis steps** between it and the target molecule), and the **fraction of the precursor molecules in that route that are in stock**.
* This gives a quantitative analysis of the **quality of the synthetic route**.

Steps 1 - 4 are then repeated in **iterative Monte Carlo simulations**. The number of iterations is governed by a **predefined limit**, or a **predefined search time**. This iterative process is illustrated in **Fig 11**.

<br>
  <div align="center">
    <img src="https://github.com/user-attachments/assets/237073ea-3bd3-4c2c-b7a0-bd779d3d09d9", alt="mcts-simulation-playout"/>
    <p>
      <b>Fig 11</b> AiZynthFinder Monte Carlo Tree Search (MCTS) steps. <b><sup>5</sup></b>
    </p>
  </div>
<br>

## 1.5 AiZynthFinder Post-Processing Tools - Route Scoring

AiZynthFinder also uses a number of **scoring algorithms** to **score routes during post-processing** (**Fig 12**).

<br>
  <div align="center">
    <img src="https://github.com/user-attachments/assets/191465e0-9f78-473f-b7f1-d0ec06fbfdb1", alt="post-processing-route-scoring-algorithms"/>
    <p>
      <b>Fig 12</b> AiZynthFinder post-processing route scoring algorithm formulae/ <b><sup>5</sup></b>
    </p>
  </div>
<br>

## 1.6 Route Clustering

AiZynthFinder also has the ability to **cluster routes** in order to perform a **cluster analysis** via **hierarchical clustering**. **<sup>11</sup>**

The specific type of hierarchical clustering that AiZynthFinder uses is **agglomerative ("bottom-up") hierarchical clustering** **<sup>9</sup>**. This involves:
1. Creating a **dendrogram** (a common visualisation toolkit from the `scipy.cluster.hierarchy` package), to represent the **hierarchy of clusters of routes formed at different levels of distance**.
2. Using a **linkage matrix** (`linkage_matrix`) to  **calculate the Euclidean distance between the clusters** at **each step of the clustering process**. This gives a **measure of similarity or dissimilarity** between the clusters of routes. This `linkage_matrix` is generated by the `ClusterHelper` class, which uses the agglomerative clustering algorithm implemented in **scikit-learn**

## 1.7 References

**[1]** Saigiridharan, L. et al. (2024) ‘AiZynthFinder 4.0: Developments based on learnings from 3 years of industrial application’, Journal of Cheminformatics, 16(1). <br><br>
**[2]** Coley, C.W. et al. (2017) ‘Prediction of organic reaction outcomes using machine learning’, ACS Central Science, 3(5), pp. 434–443. <br><br>
**[3]** Ishida, S. et al. (2022) ‘Ai-driven synthetic route design incorporated with Retrosynthesis Knowledge’, Journal of Chemical Information and Modeling, 62(6), pp. 1357–1367. <br><br>
**[4]** Zhao, D., Tu, S. and Xu, L. (2024) ‘Efficient retrosynthetic planning with MCTS Exploration Enhanced A* search’, Communications Chemistry, 7(1). <br><br>
**[5]** Genheden, S. (2022) 'AiZynthFinder', AstraZeneca R&D Presentation. Available at: https://www.youtube.com/watch?v=r9Dsxm-mcgA (Accessed: 22 August 2024). <br><br>
**[6]** Thakkar, A. et al. (2020) ‘Datasets and their influence on the development of computer assisted synthesis planning tools in the pharmaceutical domain’, Chemical Science, 11(1), pp. 154–168. <br><br>
**[7]** David, L. et al. (2020) ‘Molecular representations in AI-Driven Drug Discovery: A review and practical guide’, Journal of Cheminformatics, 12(1). <br><br>
**[8]** Genheden, S., Engkvist, O. and Bjerrum, E.J. (2020) A quick policy to filter reactions based on feasibility in AI-guided retrosynthetic planning. <br><br>
**[9]** Coulom, R. (2007) ‘Efficient selectivity and backup operators in Monte-Carlo Tree Search’, Lecture Notes in Computer Science, pp. 72–83. <br><br>
**[10]** Kroese, D.P. et al. (2014) ‘Why the monte Carlo method is so important today’, WIREs Computational Statistics, 6(6), pp. 386–392. <br><br>
**[11]** Genheden, S., Engkvist, O. and Bjerrum, E. (2021) ‘Clustering of synthetic routes using tree edit distance’, Journal of Chemical Information and Modeling, 61(8), pp. 3899–3907. <br><br>
