import pandas as pd
import numpy as np
import gensim
import sys

n_gram = 2
dimensions = 100
epochs = 100

f = open("file", "r+")
sentences = f.readlines()

csv_data = {
    'current': [],
    'next': []
}
index = 0
print("Loading sentences...")
for line in sentences:
    index += 1
    line = 'y '+line
    words = [elem if elem else None for elem in line.split()]
    for i in range(len(words) - n_gram):
        current = ''.join(words[i + j] + ' ' for j in range(n_gram))[:-1]
        next_word = words[i + n_gram]
        csv_data['current'].append(current)
        csv_data['next'].append(next_word)

# Data to Pandas data frame
print("Reading data from dictionary")
df = pd.DataFrame(csv_data, columns=['current', 'next'])

model = gensim.models.Word2Vec.load("trained_model")

print("Preparing data for model")
current_inputs = []
next_inputs = []
for index, row in df.iterrows():
    sys.stdout.write("\rPreparing line %i..." % index)
    sys.stdout.flush()
    try:
        matrix = np.array([model.wv.get_vector(word) for word in row['current'].split()])
        current_vector = matrix.flatten()
        next_vector = model.wv.get_vector(row['next'])
        current_inputs.append(current_vector)
        next_inputs.append(next_vector)
    except KeyError as e:
        print(e)
        print("Omiting line %d" % index)
        continue

# Creating neural network model
nn = Sequential()
nn.add(Dense(210, input_dim=dimensions * n_gram))
nn.add(Dense(150, activation='relu'))
nn.add(Dense(dimensions, activation='relu'))
nn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

xs = np.array(current_inputs)
ys = np.array(next_inputs)
print("%d epochs..." % epochs)
history = nn.fit(xs, ys, epochs=epochs, batch_size=10)
print(history)
model_json = nn.to_json()

# Saving neural network
with open('nn.json', "w") as json_file:
    json_file.write(model_json)

nn.save_weights('nn.h5')

def predict(sentence):
    sentence = sentence.lower()
    words_in_sentence = [elem if elem else None for elem in sentence.split()]
    if len(words_in_sentence) < n_gram:
        sentence = 'y '+sentence
    # TODO: Check if number of words is smaller than n_gram
    words_in_sentence = [elem if elem else None for elem in sentence.split()]
    last_words = words_in_sentence[-n_gram:]
    last_words_vector_list = []
    for word in last_words:
        last_words_vector_list.append(model.wv.get_vector(word))
    last_words_vector = np.array(last_words_vector_list).flatten()
    array_to_predict = np.array([last_words_vector])
    predicted_vector = nn.predict(array_to_predict, verbose=0)
    predicted_words = model.wv.similar_by_vector(predicted_vector[0], restrict_vocab=900)
