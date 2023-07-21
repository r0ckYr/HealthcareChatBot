import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import json

# Load and preprocess the training data
stemmer = LancasterStemmer()
words = []
labels = []
docs_x = []
docs_y = []

# Load the intents JSON file
with open('intents.json') as file:
    data = json.load(file)

for intent in data['intents']:
    for pattern in intent['patterns']:
        tokenized_words = nltk.word_tokenize(pattern)
        words.extend(tokenized_words)
        docs_x.append(tokenized_words)
        docs_y.append(intent['tag'])

    if intent['tag'] not in labels:
        labels.append(intent['tag'])

words = [stemmer.stem(word.lower()) for word in words if word != '?']
words = sorted(list(set(words)))

labels = sorted(labels)

training = []
output = []

for x, doc in enumerate(docs_x):
    bag = []
    tokenized_words = [stemmer.stem(word.lower()) for word in doc]
    for word in words:
        bag.append(1) if word in tokenized_words else bag.append(0)

    output_row = [0] * len(labels)
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)

training = np.array(training)
output = np.array(output)

# Convert training and output data to PyTorch tensors
training = torch.tensor(training, dtype=torch.float32)
output = torch.tensor(output, dtype=torch.float32)

# Define the neural network model using PyTorch
class ChatbotModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ChatbotModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return self.softmax(x)

input_size = len(training[0])
hidden_size = 8
output_size = len(output[0])

# Instantiate the model
model = ChatbotModel(input_size, hidden_size, output_size)

# Define loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 1100
batch_size = 16
for epoch in range(epochs):
    permutation = torch.randperm(training.size()[0])
    for i in range(0, training.size()[0], batch_size):
        indices = permutation[i:i + batch_size]
        batch_inputs, batch_outputs = training[indices], output[indices]

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(batch_inputs)
        loss = criterion(outputs, batch_outputs)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}/{epochs}, Loss: {loss.item()}")

# Save the model
torch.save(model.state_dict(), 'model.pth')

def process_input(user_input):
    tokenized_input = nltk.word_tokenize(user_input)
    tokenized_input = [stemmer.stem(word.lower()) for word in tokenized_input]

    input_bag = [0] * len(words)
    for word in tokenized_input:
        for i, w in enumerate(words):
            if w == word:
                input_bag[i] = 1

    input_bag = torch.tensor(input_bag, dtype=torch.float32)
    input_bag = input_bag.unsqueeze(0)  # Add a batch dimension

    results = model(input_bag)
    results_index = torch.argmax(results).item()
    tag = labels[results_index]

    for intent in data['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])

print("Healthcare Chatbot - Ask me anything!")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'quit':
        break

    response = process_input(user_input)
    print("Bot: " + response)

