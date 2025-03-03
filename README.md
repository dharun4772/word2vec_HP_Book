# Word2Vec on Harry Potter Books

## Project Overview
This project explores training Word2Vec models on the entire Harry Potter book series. The goal is to analyze word embeddings generated using different Word2Vec implementations presented in the original paper by Mikolov et al.

## Implementations Explored
I have experimented with all the major implementations of Word2Vec discussed in the paper:
1. **Continuous Bag of Words (CBOW)** – Predicts the target word based on surrounding context words.
2. **Skip-gram Model** – Predicts surrounding words given a target word, suitable for learning rare word representations.
3. **Hierarchical Softmax** – An efficient approach for optimizing Word2Vec training, particularly useful for large vocabularies.
4. **Negative Sampling** – A simplified training technique that helps improve efficiency by updating only a subset of weights per training step.
5. **Subword Information (FastText-like Approach)** – Enhances representations for rare and out-of-vocabulary words by considering character-level n-grams.

## Dataset
- The dataset consists of all seven books of the *Harry Potter* series.
- Text preprocessing includes:
  - Lowercasing
  - Removing special characters and punctuation
  - Tokenization
  - Stopword removal (optional, depending on the experiment)

## Model Training
- Implemented using Gensim's Word2Vec library.
- Experimented with different hyperparameters such as:
  - Vector size (100, 200, 300 dimensions)
  - Window size (context window around a word)
  - Minimum word frequency threshold
  - Number of negative samples
  - Epochs for training

## Evaluation
- **Word Similarity:** Checked cosine similarity between key terms (e.g., "Harry", "Voldemort", "magic").
- **Analogy Tasks:** Tested model predictions on relationships (e.g., "Hogwarts" is to "school" as "Ministry of Magic" is to "government").
- **TSNE Visualization:** Plotted word embeddings to visualize clustering of related words.

## Findings
- Skip-gram performed better for learning relationships between rare words, which was important given the unique names in *Harry Potter*.
- Negative sampling improved computational efficiency without significant loss in accuracy.
- Subword information helped with character names and spell-related words.
- CBOW was faster but sometimes struggled with rare words.

## Future Work
- Experimenting with contextual embeddings (e.g., BERT, ELMo) to compare performance.
- Fine-tuning hyperparameters further to improve clustering and analogy tasks.
- Using other fantasy book series as datasets to generalize findings.

## Requirements
To run the experiments, install the following dependencies:
```bash
pip install gensim nltk matplotlib seaborn
```

## Running the Code
```python
from gensim.models import Word2Vec
# Load preprocessed data and train model
model = Word2Vec(sentences, vector_size=200, window=5, min_count=5, workers=4, sg=1)
# Save model
model.save("word2vec_hp.model")
```

## Conclusion
Training Word2Vec on *Harry Potter* books provided interesting insights into how embeddings capture relationships between magical concepts, characters, and locations. Different implementations have their strengths, and the choice depends on the desired outcome and computational constraints.

