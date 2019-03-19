# importing libraries
import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Load Data Function

    Arguments:
        messages_filepath -> filepath to the messages.csv file
        categories_filepath -> filepath to the categories.csv file
    Returns:
        df -> A merged Pandas DataFrame of messages and categories
    '''
    # loading messages dataset
    messages = pd.read_csv(messages_filepath)
    # loading categories dataset
    categories = pd.read_csv(categories_filepath)
    # merging datasets
    df = messages.merge(categories, how = 'outer', on='id')

    return df


def clean_data(df):
    '''
    Clean Data Function

    Arguments:
        df -> Pandas DataFrame of messages and categories
    Returns:
        df -> A processed and clean version of original DataFrame
    '''
    # creating a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(pat=';',expand=True)
    # selecting the first row of the categories dataframe
    row = categories.iloc[0,:]
    # appling a lambda function that takes everything
    # up to the second to last character of each string with slicing
    category_colnames = list(row.apply(lambda x: x.split('-')[0]))
    # renaming the columns of `categories`
    categories.columns = category_colnames

    # For loop to iterate through columns in categories
    for column in categories:
        # setting each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x.split('-')[1])
        # converting column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    # droping the original categories column from `df`
    df.drop(['categories'], axis = 1, inplace=True)
    # concatenating the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # droping duplicates in our DataFrame
    df = df[df.duplicated() == False]

    # Dropping relevant rows
    df = df.dropna(axis=0, thresh = 1, subset = list(df.columns)[4:] )


    return df


def save_data(df, database_filename):
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('dfTable', engine, if_exists="replace")


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
