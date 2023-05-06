import requests
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import spacy
from collections import Counter
import os
import random
import nltk
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
nltk.download('punkt')

nlp = spacy.load("en_core_web_sm")

book_urls = [
    "https://www.gutenberg.org/files/1342/1342-0.txt",
    "https://www.gutenberg.org/files/345/345-0.txt",
    "https://www.gutenberg.org/files/76/76-0.txt",
    "https://www.gutenberg.org/files/1260/1260-0.txt",
    "https://www.gutenberg.org/files/1661/1661-0.txt",
    "https://www.gutenberg.org/files/768/768-0.txt",
    "https://www.gutenberg.org/files/161/161-0.txt",
    "https://www.gutenberg.org/files/84/84-0.txt",
    "https://www.gutenberg.org/files/514/514-0.txt",
    "https://www.gutenberg.org/files/98/98-0.txt",
    "https://www.gutenberg.org/files/1400/1400-0.txt",
    "https://www.gutenberg.org/files/2701/2701-0.txt",
    "https://www.gutenberg.org/files/174/174-0.txt",
    "https://www.gutenberg.org/files/1184/1184-0.txt",
    "https://www.gutenberg.org/files/135/135-0.txt",
]

text = ""
for book_url in book_urls:
    response = requests.get(book_url, stream=True)
    for line in response.iter_lines():
        if line:
            text += line.decode("utf-8").strip()


tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(text)


np.save("char_to_int.npy", tokenizer.word_index)

sentences = []
for book_url in book_urls:
    response = requests.get(book_url, stream=True)
    for line in response.iter_lines():
        if line:
            
            tokens = nltk.word_tokenize(line.decode('utf-8').strip())
            
            sentences.append(' '.join(tokens))


random.shuffle(sentences)


train_size = int(0.8 * len(sentences))
train_sentences = sentences[:train_size]
test_sentences = sentences[train_size:]

tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts([sentence for sentence in train_sentences])

max_length = max([len(tokenizer.texts_to_sequences([sentence])[0]) for sentence in sentences])

def data_generator(sentences, tokenizer, batch_size, max_length):
    
    while True:

        random.shuffle(sentences)

        for i in range(0, len(sentences), batch_size):
            batch_sentences = sentences[i:i+batch_size]
            batch_sequences = tokenizer.texts_to_sequences(batch_sentences)
            batch_padded_sequences = pad_sequences(batch_sequences, maxlen=max_length, padding="post")
            batch_target = [to_categorical(seq, num_classes=len(tokenizer.word_index)+1) for seq in batch_sequences]
            batch_target = pad_sequences(batch_target, maxlen=max_length, padding="post")
            batch_target = batch_target.reshape(batch_target.shape[0], batch_target.shape[1], len(tokenizer.word_index)+1)

            yield batch_padded_sequences, batch_target

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=64, input_length=max_length),
    tf.keras.layers.Bidirectional
    (tf.keras.layers.LSTM(64, return_sequences=True)),
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(len(tokenizer.word_index)+1, activation='softmax'))
])

model.compile(loss='categorical_crossentropy', optimizer='adam')

batch_size = 32
train_steps_per_epoch = len(train_sentences) // batch_size
test_steps_per_epoch = len(test_sentences) // batch_size

train_generator = data_generator(train_sentences, tokenizer, batch_size, max_length)
test_generator = data_generator(test_sentences, tokenizer, batch_size, max_length)

model.fit(train_generator, epochs=1, steps_per_epoch=train_steps_per_epoch, validation_data=test_generator, validation_steps=test_steps_per_epoch)
model.save('book_test_3.h5')



