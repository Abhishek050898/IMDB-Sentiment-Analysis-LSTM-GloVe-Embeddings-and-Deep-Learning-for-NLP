
# Sentiment Analysis using LSTM on IMDB Movie Reviews

## Project Overview

This project focuses on **Sentiment Analysis** of movie reviews from the IMDB dataset using **Deep Learning**. The dataset consists of 50,000 reviews labeled as either positive or negative. A deep learning model is built using the **LSTM** (Long Short-Term Memory) architecture to classify the sentiments of the reviews.

### Key Features

- **Text Preprocessing**: The project involves text cleaning and preprocessing steps such as removal of HTML tags, punctuation, and stopwords.
- **Word Embeddings**: Pre-trained **GloVe embeddings** are used to capture the semantic meaning of words in the reviews.
- **Deep Learning Model**: The model employs an **LSTM** architecture to process the sequential nature of text data, with additional layers like **Conv1D** and **Dense**.
- **Sentiment Classification**: The model is trained to predict whether a review is positive or negative based on the textual content.

## Installation

### Requirements

To run this project, you'll need to install the following dependencies. You can install them by running:

```bash
pip install -r requirements.txt
```

### Dataset

The IMDB dataset used in this project can be downloaded from [Kaggle](https://www.kaggle.com/). Make sure to place the dataset file `IMDB Dataset.csv` in the project directory.

### Running the Project

1. **Install dependencies**: Ensure you have installed the required libraries using the `requirements.txt`.
2. **Download pre-trained GloVe embeddings**: You need the GloVe vectors (e.g., `glove.6B.100d.txt`) which can be downloaded from [here](https://nlp.stanford.edu/projects/glove/).
3. **Run the Jupyter Notebook**: Open `imdb.ipynb` and run the cells step-by-step to preprocess the data, build the model, and train it.

### Model Architecture

The deep learning model consists of the following layers:
- **Embedding Layer**: Initialized with pre-trained GloVe vectors.
- **Convolutional Layer (Conv1D)**: Used to extract feature representations from the embeddings.
- **LSTM Layer**: Captures long-term dependencies within the reviews.
- **Dense Layer**: Outputs the sentiment classification (positive or negative).

## Results

After training, the model achieves high accuracy in predicting the sentiment of unseen movie reviews, demonstrating the effectiveness of LSTM for text classification tasks.
