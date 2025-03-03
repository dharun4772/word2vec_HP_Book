# -*- coding: utf-8 -*-
"""harrypotter-word2vec.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1sW0e0YK9bmu0OrwrC5sH1j6I6NcBSsb_
"""

# Commented out IPython magic to ensure Python compatibility.
# %pip install -qU contractions

import numpy as np
import pandas as pd
import os
import string
import contractions
import random
import tensorflow as tf
import keras
import string
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import skipgrams

import kagglehub

# Download latest version
path = kagglehub.dataset_download("shubhammaindola/harry-potter-books")

print("Path to dataset files:", path)

contents = os.listdir(path)
for item in contents:
    print(item)

overall_text = ""
for item in contents:
    with open(f"{path}/{item}") as f:
        text = f.read()
    overall_text +="\n"+text

len(overall_text)

import nltk
from nltk.corpus import stopwords
nltk.download("stopwords")

def preprocess(text):
  text = text.lower()
  replacer = str.maketrans('','',string.punctuation)
  text = text.translate(replacer)
  text = contractions.fix(text)
  text = text.replace(",","")
  text = text.replace("—","")
  text = text.replace("-","")
  text = text.replace("'","")
  text = text.replace("\"","")
  text = text.replace("”","")
  text = text.replace("“","")
  text = text.replace("’","")
  return text

overall_text = preprocess(overall_text)
overall_words = len(overall_text.split())
sub_sampling_d = {}
for word in overall_text.split():
    if word not in sub_sampling_d:
        sub_sampling_d[word] = [1]
    else:
        sub_sampling_d[word][0] += 1

threshold = 1e-4
for word in sub_sampling_d:
    sub_sampling_d[word].append(1 - np.sqrt(threshold/((sub_sampling_d[word][0])/overall_words)))

def sub_sampleword(word):
  thres = sub_sampling_d[word][1]
  return random.random() > thres-0.05

filtered_text = " ".join([word for word in overall_text.split() if sub_sampleword(word)])

def faster_phrasal_extraction(filtered_text, overall_text):
    vocab = set(filtered_text.split())
    initial_vocab_size = len(vocab)
    print(f"Initial vocabulary size: {initial_vocab_size}")

    word_counts = {}
    words = overall_text.split()
    for word in words:
        word_counts[word] = word_counts.get(word, 0) + 1

    bigram_counts = {}
    for idx in range(len(words) - 1):
        if idx % 10000 == 0:
            print(f"Computing bigrams: {idx}/{len(words)}")

        if '\n' in words[idx:idx+2] or ' ' in words[idx:idx+2]:
            continue

        bigram = (words[idx], words[idx+1])
        bigram_counts[bigram] = bigram_counts.get(bigram, 0) + 1

    delta = 4
    added_phrases = 0

    for bigram, count_01 in bigram_counts.items():
        count_0 = word_counts.get(bigram[0], 0)
        count_1 = word_counts.get(bigram[1], 0)
        score = (count_01 - delta) / (count_0 * count_1) if count_0 * count_1 > 0 else 0

        if score > 0.001:
            bigram_str = f"{bigram[0]} {bigram[1]}"
            vocab.add(bigram_str)
            added_phrases += 1

        if added_phrases % 1000 == 0 and added_phrases > 0:
            print(f"Added {added_phrases} phrases so far")

    print(f"Final vocabulary size: {len(vocab)}")
    print(f"Added {added_phrases} phrases to vocabulary")

    return vocab

vocab =  faster_phrasal_extraction(filtered_text, overall_text)

vocab_d = {}
for i, val in enumerate(vocab):
    vocab_d[val] = i

len(vocab)

skip_grams = []
context_size = 6
ov = overall_text.split()
for i in range(1, len(ov)-1):
    if ov[i]+" "+ov[i+1] in vocab_d:
      inp = vocab_d[ov[i]+" "+ov[i+1]]
      context = []
      for j in range(context_size):
          if i-j+1>=0:
              context.append(vocab_d[ov[i-j+1]])
          if i+j+2<=context_size:
              context.append(vocab_d[ov[i+j+2]])
      for w in context:
          skip_grams.append([inp, w])
    elif ov[i] in vocab_d:
      inp = vocab_d[ov[i]]
      context = []
      for j in range(context_size):
          if i-j+1>=0:
              context.append(vocab_d[ov[i-j+1]])
          if i+j+1<=context_size:
              context.append(vocab_d[ov[i+j+1]])
      for w in context:
          skip_grams.append([inp, w])

skip_grams

np.random.seed(seed=42)

def random_batch(data,size):
    random_inputs=[]
    random_labels=[]
    random_index=np.random.choice(range(len(data)),size, replace=False)
    for i in random_index:
        one = np.zeros(len(vocab))
        one[data[i][0]]=1
        random_inputs.append(one)
        random_labels.append(data[i][1])
    return random_inputs, random_labels
test = random_batch(skip_grams[:6], size=3)

from keras.models import Sequential
from keras.layers import Embedding, Dense, Flatten
from keras.optimizers import Adam

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dot, Flatten


embedding_dim = 100
window_size = 1

target_input = Input(shape=(1,), name="target_input")
context_input = Input(shape=(1,), name="context_input")

target_embedding = Embedding(input_dim=len(vocab), output_dim=embedding_dim, name="target_embedding")(target_input)
context_embedding = Embedding(input_dim=len(vocab), output_dim=embedding_dim, name="context_embedding")(context_input)

dot_product = Dot(axes=2)([target_embedding, context_embedding])
output = Flatten()(dot_product)

model = Model(inputs=[target_input, context_input], outputs=output)
model.compile(optimizer="adam", loss="binary_crossentropy")

# Print model summary
model.summary()

target_words, context_words = zip(*skip_grams)
target_words = np.array(target_words, dtype="int32")
context_words = np.array(context_words, dtype="int32")

labels = np.ones(len(target_words))

model.fit(
    [target_words, context_words],
    labels,
    epochs=10,
    batch_size=64,
    verbose=1
)

model.save("ownword2vech.keras")

# prompt: check for random 10 words the closest embedding word based on dot product and also plot the 2d of some sampled vectors

import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Assuming 'model' and 'vocab_d' are defined from the previous code

# Get the embedding layer weights
embedding_weights = model.get_layer("target_embedding").get_weights()[0]

# Function to find the closest word embedding
def find_closest_embedding(word, embedding_weights, vocab_d):
    if word not in vocab_d:
        return "Word not in vocabulary"

    word_index = vocab_d[word]
    word_embedding = embedding_weights[word_index]

    dot_products = np.dot(embedding_weights, word_embedding)
    closest_index = np.argsort(dot_products)[::-1][1] # Exclude the word itself
    closest_word = [key for key, value in vocab_d.items() if value == closest_index][0]
    return closest_word


# Example usage for 10 random words
random_words = random.sample(list(vocab_d.keys()), 10)

for word in random_words:
    closest = find_closest_embedding(word, embedding_weights, vocab_d)
    print(f"Word: {word}, Closest word: {closest}")


# Function to plot 2D representation of word embeddings
def plot_embeddings(embedding_weights, vocab_d, num_samples=100):
    # Use t-SNE to reduce dimensionality to 2D
    tsne = TSNE(n_components=2, random_state=42)
    sampled_indices = random.sample(range(len(vocab_d)), num_samples)
    sampled_embeddings = embedding_weights[sampled_indices]
    reduced_embeddings = tsne.fit_transform(sampled_embeddings)

    # Plot the embeddings
    plt.figure(figsize=(10, 8))
    for i, embedding in enumerate(reduced_embeddings):
        word = [key for key, value in vocab_d.items() if value == sampled_indices[i]][0]
        plt.scatter(embedding[0], embedding[1])
        plt.annotate(word, (embedding[0], embedding[1]), fontsize=8)

    plt.title("2D Word Embeddings Visualization")
    plt.show()


# Example usage with 100 samples
plot_embeddings(embedding_weights, vocab_d, num_samples=1000)