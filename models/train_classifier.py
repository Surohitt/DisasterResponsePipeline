# import libraries
import sys
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger','stopwords'])
from nltk.corpus import stopwords
import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pickle
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sqlalchemy import create_engine
from sklearn.metrics import classification_report
from sklearn.multioutput import MultiOutputClassifier


def load_data(database_filepath):
    # connect to the database
    engine = create_engine('sqlite:///'+ database_filepath)
    # load data from database
    df = pd.read_sql_table('dfTable', engine)
    # Splitting the data into predictors and targets
    X = df.message
    Y = df[list(df.columns[5:])]
    # Mapping the '2' values in 'related' to '1' - because they are considered as a response
    Y['related']=Y['related'].map(lambda x: 1 if x == 2 else x)
    # category names
    category_names = Y.columns
    return X, Y, category_names


def tokenize(text):
    """
    Tokenize function

    Arguments:
        text -> text from messages
    Output:
        clean_tokens -> returns tokenized text
    """
    # Converting all text to lower case
    text=text.lower()

    # Removing punctuation from all text
    text = re.sub(r"[^a-zA-Z0-9]"," ",text)
    words = word_tokenize(text)

    # Removing stop words
    new_words = []
    for w in words:
        if w not in stopwords.words('english'):
            new_words.append(w)

    # Lemmatizing our new words
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for word in new_words:
        clean_tok = lemmatizer.lemmatize(word).strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    Build Model Function

    Builds the pipeline skeleton to be used for model fitting and predicting
    """
    model = Pipeline([('cvect', CountVectorizer(tokenizer = tokenize)),
                     ('tfidf', TfidfTransformer()),
                     ('clf', RandomForestClassifier(max_depth = None, min_samples_leaf = 2, min_samples_split = 3, n_estimators = 15))
                     ])
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate Model Function

    Arguments:
        model -> model object
        X_test -> our predictor testing data / text data
        Y_test -> correctly classified target variables
        category_names -> all classification labels for target values
    Output:
        Metrics of model performance
    """

    # Predicting our model outputs on the testing data
    y_pred_test = model.predict(X_test)
    # Creating a pandas dataframe from our prediction
    y_pred_pd = pd.DataFrame(y_pred_test, columns = category_names)
    # Printing a evaluation table for each column
    for column in Y_test.columns:
        print('------------------------------------------------------\n')
        print('FEATURE: {}\n'.format(column))
        print(classification_report(Y_test[column],y_pred_pd[column]))

    # Printing the overall accuracy of our model
    overall_accuracy = (y_pred_test == Y_test).mean().mean()
    print('\n\nAverage overall accuracy {0:.2f}% \n'.format(overall_accuracy*100))



    pass


def save_model(model, model_filepath):
    '''
    Save Model function that saves the model to a particular filepath

    Arguments:
        model -> ML model
        model_filepath -> Where we want to store the model
    Output:
        model stored in particular location
    '''

    filename = model_filepath
    pickle.dump(model, open(filename, 'wb'))
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
