# HealthcareChatBot
## Healthcare Chatbot

This is a simple healthcare chatbot that can answer user queries related to healthcare topics. The chatbot is built using Python and PyTorch for natural language processing and deployed as a Flask application on Amazon EC2.

## Features
The chatbot uses a bag-of-words model and a feedforward neural network to understand user queries and provide appropriate responses.
It can handle multiple intents (queries) and respond accordingly based on the intent recognition.
The chatbot is trained on a dataset stored in a JSON file, which contains user patterns and corresponding intent tags.
Getting Started
To run the chatbot locally or deploy it on your own Amazon EC2 instance, follow the steps below:

## Prerequisites
Python 3.6 or above
PyTorch
Flask
NLTK

## Installation

1. Clone the repository to your local machine or EC2 instance
```
git clone https://github.com/r0ckYr/HealthcareChatBot.git
cd HealthcareChatBot
```
2. Install the required dependencies using pip
```
pip install -r requirements.txt
```

## Usage
1. Training the Chatbot (Optional)
If you want to retrain the chatbot on your custom dataset, follow these steps:
Create or modify the 'intents.json' file with your dataset, following the existing structure.
Run the training script:
```
python train_chatbot.py
```

2. Starting the Flask Application To run the chatbot as a Flask application, execute the following command:
```
python app.py
```

The Flask application will be running on http://localhost:5000/ or the public IP of your EC2 instance.
