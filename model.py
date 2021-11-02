import functions as mf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.tokenize import TreebankWordTokenizer 
import nltk   
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from imblearn.over_sampling import RandomOverSampler, SMOTE
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split


# setting global variables
RANDOM_STATE = 42


def check_non_ascii(df):
    '''
    This takes a dataframe and cleans the lyrics column of the dataframe of non ascii characters 
    and returns the cleaned dataframe
    '''
    for i in range(len(df.lyrics)):
        encoded_string = df['lyrics'][i].encode("ascii", "ignore")
        df['lyrics'][i] = encoded_string.decode()
    return df



def corpus_maker(df):
    '''
    This takes a dataframe, transforms the texts in the lyrics columns into a list and makes all lower case
    and returns the corpus as a list of lyrics
    '''
    CORPUS = df['lyrics'].tolist()
    CORPUS = [s.lower() for s in CORPUS]
    return CORPUS



def corpus_cleaner(CORPUS):
    '''
    This takes the corpus and tokanizes and lemmatizes it
    and returns the cleaned corpus list
    '''
    CLEAN_CORPUS = []
    tokenizer = TreebankWordTokenizer()
    lemmatizer = WordNetLemmatizer()

    for doc in CORPUS:
        tokens = tokenizer.tokenize(text=doc)
        clean_doc = " ".join(lemmatizer.lemmatize(token) for token in tokens)
        CLEAN_CORPUS.append(clean_doc)
    return CLEAN_CORPUS
    


def creating_labels(artist_1, artist_2, num_of_songs):
    '''
    This takes two strings for a artist names and the number of songs as integer
    and duplicates the artist names 
    and returns the label as a list
    '''
    LABELS = [artist_1] * num_of_songs + [artist_2] * num_of_songs
    return LABELS



def vectorize_and_normalize_corpus(CLEAN_CORPUS, LABELS):
    '''
    This takes the cleaned corpus as a list 
    instanciates the vectorizer which attributes every word in the cleaned corpus to a binary vector
    and returns a dataframe with normalized entries
    '''
    vectorizer = CountVectorizer() 
    vectors = vectorizer.fit_transform(CLEAN_CORPUS)
    tf = TfidfTransformer() 
    vectors_normalized = tf.fit_transform(vectors)
    df_vec_norm = pd.DataFrame(vectors_normalized.todense(), columns=vectorizer.get_feature_names(), index=LABELS)
    return df_vec_norm
    


def vectorize_words(CLEAN_CORPUS):
    '''
    This takes the cleaned corpus as a list and instatiates the vectorizer and apply the vecotrizer on the cleaned corpus
    and returns X
    '''
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(CLEAN_CORPUS)
    return X


def split_data(X, y):
    '''
    This takees the feature data and label data and splits it
    '''
    return train_test_split(X, y, random_state=RANDOM_STATE, stratify=y)
    


def ml_model(CLEAN_CORPUS, LABELS, text, preprocessing = TfidfVectorizer()):
    '''
    This takes the cleaned corpus and labels applies a pipeline with Vectorizer(Transformer) and machine learning algorithm (e.g. logistic regression)
    and returns the predicted band and the prediction propability 
    
    '''
    steps = [('tf-idf', preprocessing),
         
              #('LR', LogisticRegression()),
              ('MNB', MultinomialNB())
            ]

    pipeline = Pipeline(steps)
    pipeline.fit(CLEAN_CORPUS, LABELS)

    print(f'Your text is more {pipeline.predict([text])[0]} like with a probability ratio of {pipeline.predict_proba([text])[0]}.')


def print_evaluations(ytrue, ypred, model):
    print(f'How does model {model} score:')
    print(f'The accuracy of the model is: {round(metrics.accuracy_score(ytrue, ypred), 3)}')
    print(f'The precision of the model is: {round(metrics.precision_score(ytrue, ypred), 3)}')
    print(f'The recall of the model is: {round(metrics.recall_score(ytrue, ypred), 3)}')
    print(f'The f1-score of the model is: {round(metrics.f1_score(ytrue, ypred), 3)}')
    
    #print confusion matrix
    fig = plt.figure(figsize=(6, 6))
    cm = metrics.confusion_matrix(ytrue, ypred)
    print(cm)
    
    #plot the heatmap
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax); #annot=True to annotate cells
    
    # labels, title and ticks
    ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
    ax.set_title('Confusion Matrix'); 
    ax.xaxis.set_ticklabels(['Artist 1', 'Artist 2']); 
    ax.yaxis.set_ticklabels(['Artist 1', 'Artist 2'])




if __name__ == "__main__":
    pass    