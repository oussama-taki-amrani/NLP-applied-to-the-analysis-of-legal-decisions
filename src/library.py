import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

# ne pas afficher les warnings
import warnings
warnings.filterwarnings("ignore")

NOM = "NOM"


# fonction given a sentence and a a list of words return true if the sentence contains one of the words
def contains_word(sentence, words):
    sentence_words = sentence.split(" ")
    for word in words:
        if word in sentence_words:
            return True
    return False



# fonction qui supprimer les lignes vides d'un text
def remove_empty_lines(text):
    # split the text into lines
    lines = text.splitlines()
    # remove empty lines
    lines = [line for line in lines if line.strip() != '']
    # join the lines again
    text = "\n".join(lines)
    return text



def clean_text(text):
    # Enlever les espaces en début et fin de ligne
    text = remove_empty_lines(text)
    # Enlever les chiffres
    text = re.sub(r"\d+", "", text)
    # lower case
    text = text.lower()
    allowed_characters = r"[^a-zA-Z0-9ÀÁÂÃÄÅàáâãäåÒÓÔÕÖØòóôõöøÈÉÊËèéêëÇçÌÍÎÏìíîïÙÚÛÜùúûüÿÑñ,.'’;:-_\n\s]"
    text = re.sub(allowed_characters, '', text)
    # eliminer les espaces multiples
    text = re.sub(' +', ' ', text)
    # enlever les saut multiple de ligne
    text = re.sub(r'\n+', '.', text)
    # remplacer un caractere suivi de trois points par un Nom arbitraire
    text = re.sub(r'\w\.\.\.', 'NOM ', text)
    # supprimer les mot qui contiennent moins 1 caractère
    text = re.sub(r'\b\w{1}\b', '', text)
    # remplace les apostrophes par un espace
    text = re.sub(r'’', ' ', text)
    
    return text

def custom_sent_tokenize(text):
    # Define the regular expression to match end-of-sentence punctuation 
    pattern = r'([.!?\n])'
    
    # Tokenize the text using the regular expression 
    sentences = re.split(pattern, text)
    # trash the void sentences or sentences with only spaces
    for i in range(len(sentences)):
        if len(sentences) == 0  or len(sentences) == 1:
            sentences.pop(i)
    
    return sentences




def remove_stopwords(text):
    stop_words = set(stopwords.words("french"))
    # tokenize the text
    words = text.split()
    # remove stop words except il and elle
    words = [word for word in words if word not in stop_words or word == "il" or word == "elle"]
    # join words to make text again
    text = " ".join(words)
    return text

key_words_to_avoid = ['condamné','condamnée','coupable','condamner','fautif','vol']
key_words = ['préjudices','péjudice','souffrance','nécessite','nécessité','payer','l’indemnisation','hospitalisation','contamination','victime','indemniser','indemnisation','accident','blessé','subi','blessure','violé','attaqué','violée','attaquer','attaquée','feu','arme','dégats','versées','versés','perte','souffrance','décès','mort','mortel','mortelle','mortellement','mortel','handicap','subi','subir','subie','lésions']
necessary_words = ['il','elle','monsieur','madame','melle','m','mme']

## main function to process the text
def process_text_to_sentences(text):
    
    # clean the text
    text = clean_text(text)
    # write text in a file
    sentences = custom_sent_tokenize(text)

    ## preprocess the sentences

    # remove stopwords expect il and elle 
    #sentences = [remove_stopwords(sentence) for sentence in sentences]
    # remove ponctuation and numbers except virgule, point
    sentences = [re.sub(r"[^a-zA-Z0-9ÀÁÂÃÄÅàáâãäåÒÓÔÕÖØòóôõöøÈÉÊËèéêëÇçÌÍÎÏìíîïÙÚÛÜùúûüÿÑñ,\n\s]", '', sentence) for sentence in sentences]
    # split the sentences with a comma or a point into sentences
    sentences = [sentence.split(",") for sentence in sentences]
    # plane the list
    sentences = [item for sublist in sentences for item in sublist]
    # eliminate first space
    sentences = [sentence.lstrip() for sentence in sentences]
     # remove multiple spaces
    sentences = [re.sub(' +', ' ', sentence) for sentence in sentences]

    # sentences extraction
    # remove all sentences that don't contain a name in the list ## important !!!!
    

    cleaned_sentences = []
    for i in range(len(sentences)):
        if contains_word(sentences[i], key_words):
            cleaned_sentences.append(sentences[i])    
    sentences = cleaned_sentences

    # remove sentences that don't contain a key_word_to_avoid
    cleaned_sentences = []
    for i in range(len(sentences)):
        if not contains_word(sentences[i], key_words_to_avoid):
            cleaned_sentences.append(sentences[i])
    sentences = cleaned_sentences

    # remove sentences that don't contain a necessary_word
    cleaned_sentences = []
    for i in range(len(sentences)):
        if contains_word(sentences[i], necessary_words):
            cleaned_sentences.append(sentences[i])
    sentences = cleaned_sentences

    # remove th doublon sentences
    sentences = list(dict.fromkeys(sentences))
    

    # remove sentences with only one word or two
    sentences = [sentence for sentence in sentences if len(sentence.split()) > 3]
    # remove empty sentences
    sentences = [sentence for sentence in sentences if sentence != '']

    return sentences

# create a bag of words for each sentence a bag of words is a vector of 0 and number of words in the sentence
def tokenize_fr(text):
    return word_tokenize(text, language='french')



# open all the files in the folder
ouput=  open("sentences_from_all_files" , "w", encoding="utf-8")
files = open("train_folder_predilex\\train_folder\\x_train_ids.csv", "r",encoding="utf8")
# each line is a file name with number id with uft8 encoding
list_file_names = files.readlines()
# eliminate until the first comma
list_file_names = [line.split(",")[1] for line in list_file_names]
# eliminate the \n
list_file_names = [line.strip() for line in list_file_names]
# eliminate the first line
list_file_names.pop(0)



# preparing the data frame for training



PATH = "train_folder_predilex\\train_folder\\txt_files\\"
for i in range(len(list_file_names)):
    list_file_names[i] = PATH + list_file_names[i] 


def fill_df(list_file_names):
    
    #create a dataframe
    df = pd.DataFrame(columns=["textId","sentences","Bag of words","labels"])
    i = 0     
    all_sentences = []
    empty = ['empty']
    for file in list_file_names:
        # open the file
        with open( file, "r", encoding="utf-8") as f:
            text = f.read()
        # process the text
        sentences = process_text_to_sentences(text)

        # if the sentences list is empty fill the dataframe with a empty list
        if len(sentences) == 0:
            df = pd.concat([df, pd.DataFrame({"textId":i,"sentences":empty,"labels":0})], ignore_index=True)

        # add the sentences to the dataframe using concat
        else:
            df = pd.concat([df, pd.DataFrame({"textId":i,"sentences":sentences})], ignore_index=True)

        # for vectorization
        all_sentences.append(sentences)            
        i+=1
        # write the sentences in a file
        for sentence in sentences:
            ouput.write(sentence)
            ouput.write("\n")

    all_sentences = df["sentences"].tolist()
    vectorizer = CountVectorizer(tokenizer=tokenize_fr)
    X = vectorizer.fit_transform(all_sentences)
    bagOfWords = X.toarray()
    for i in range(len(bagOfWords)):
        df["Bag of words"][i] = bagOfWords[i]
 
    return df

def label_extraction():
    labeled_sentences = open("labeled_sentences", "r", encoding="utf-8")
    # read the file
    lines = labeled_sentences.readlines()
    # for each line delete everything expect numbers 
    lines = [re.sub(r"[^0-9]", '', line) for line in lines]
    # for each line in the file convert the number to int
    lines = [int(line) for line in lines]
    return lines

lines = label_extraction()





df = fill_df(list_file_names)
    

df_train = df[0:188]

# fill the labels 
for i in range(len(lines)):
    df_train["labels"][i] = lines[i]


df_test = df[188:]




## wow now we have a two data frames one for training and one for testing



from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# extract the bag of words from the dataframe
def train_model(df_train):
    X = df_train["Bag of words"].tolist()
    y = lines

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    clf = SVC(kernel='linear')
    clf.fit(X_train, y_train)
    return clf


# Test the SVM model on the testing data
#y_pred = clf.predict(X_test)
#
### are important because we have a small labeled data set
## Calculate evaluation metrics
#accuracy = accuracy_score(y_test, y_pred)
#precision = precision_score(y_test, y_pred, average='macro')
#recall = recall_score(y_test, y_pred, average='macro')
#f1 = f1_score(y_test, y_pred, average='macro')

# Print the evaluation metrics 
#print("Accuracy:", accuracy)
#print("Precision:", precision)
#print("Recall:", recall)
#print("F1 score:", f1)



# print("now for regression model")
# use now a logistic regression model
#from sklearn.linear_model import LogisticRegression
#
#clf = LogisticRegression(random_state=0).fit(X_train, y_train)
#
## Test the SVM model on the testing data
#y_pred = clf.predict(X_test)
#
## Calculate evaluation metrics
#accuracy = accuracy_score(y_test, y_pred)
#precision = precision_score(y_test, y_pred, average='macro')
#recall = recall_score(y_test, y_pred, average='macro')
#f1 = f1_score(y_test, y_pred, average='macro')

# Print the evaluation metrics
#print("Accuracy:", accuracy)
#print("Precision:", precision)
#print("Recall:", recall)
#print("F1 score:", f1)



# now we are going to predict the labels for the non labeled sentences
def predict_labels(df, clf):
    # get the bag of words using the pandas tools
    X = df["Bag of words"].tolist()
    # predict the labels
    y_pred = clf.predict(X)
    # add the labels to the dataframe
    df["labels"] = y_pred
    return df








# merge the two dataframes


# now we are going to count the number of 1 and 2 for each textId


# function that count score_male and score_female 
def score(filtred_df):
    sMale = 0
    sFemale = 0
    # copy the labels column to a list
    labels = filtred_df["labels"].tolist()
    for i in range(len(labels)):
        # compare the label to 1
        if labels[i] == 1:
            sMale += 1
        # compare the label to 2
        elif labels[i] == 2:
            sFemale += 1
    return sMale,sFemale

def predict_sex(df):
    # for each textId
    predictedSex = []
    for i in range(len(df["textId"].unique())):
        filtred_df = df[df["textId"] == i]
        # if the dataframe is not empty
        sMale, sFemale = score(filtred_df)
        #print(sMale, sFemale)
        if sMale <= sFemale:
            predictedSex.append("femme")
        elif sMale > sFemale:
            predictedSex.append("homme")
    return predictedSex





# compare the two lists how many elements are the same over the total number of elements
def precision(list_pred, list_test):
    count = 0
    for i in range(len(list_pred)):
        if list_test[i] == list_pred[i] :
            count += 1
    return count/len(list_pred)



def metrics(list_predicted,list_y):
    # true positive
    tp = 0
    # true negative
    tn = 0
    # false positive
    fp = 0
    # false negative
    fn = 0
    for i in range(len(list_predicted)):
        if list_predicted[i] == "homme":
            if list_y[i] == "homme":
                tp += 1
            elif list_y[i] == "femme":
                fp += 1
        elif list_predicted[i] == "femme":
            if list_y[i] == "femme":
                tn += 1
            elif list_y[i] == "homme":
                fn += 1
    return tp,tn,fp,fn
