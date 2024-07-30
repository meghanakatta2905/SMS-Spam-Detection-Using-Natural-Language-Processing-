The task at hand is to classify text messages as either spam or ham (non-spam). We have a dataset containing text messages in multiple languages, including English, Hindi, French, and German. We aim to build a text classification system capable of handling multilingual text and effectively distinguishing between spam and non-spam messages.

Environment Setup:
Ensure Python and required libraries are installed (numpy, pandas, scikit-learn, beautifulsoup4, nltk, indic-nlp, imbalanced-learn, spacy, tensorflow, keras, seaborn, matplotlib).
Download language models for NLTK (nltk.download('punkt'), nltk.download('stopwords'), nltk.download('wordnet')).
Install langdetect (pip install langdetect).

Dataset:
Download the dataset (it's a CSV file named 'NLP.csv').
Update the file path in the code.

Code:
Data Preprocessing: Remove HTML tags from text, lowercase the text, remove punctuation, tokenization, remove stopwords, lemmatization, and combine text from all language columns.

Feature Extraction: Utilize Bag-of-Words representation using CountVectorizer.
Oversample the training data using SMOTE to handle class imbalance.

Model Building and Evaluation:
Train a Multinomial Naive Bayes classifier and evaluate accuracy, precision, recall, and F1-score.

Additional Model: Support Vector Machine (SVM):
Train a Support Vector Machine classifier to evaluate performance metrics and visualize the confusion matrix.

Additional Model: LSTM Neural Network:
Tokenize and pad sequences for input to the LSTM model.
Define an LSTM model architecture with Embedding and LSTM layers.
Train the model on the preprocessed text data.
Evaluate the model using accuracy, precision, recall, and F1-score, and visualize the confusion matrix.
Plot training and validation accuracy/loss curves.

Running the Code:
Execute the provided code in a Python environment or Jupyter Notebook.
Ensure all code cells are executed in sequence for proper functioning.
Check the output for accuracy, precision, recall, F1-score, and confusion matrices for each model.

Use the pred(text) function to predict the label of new text inputs.

