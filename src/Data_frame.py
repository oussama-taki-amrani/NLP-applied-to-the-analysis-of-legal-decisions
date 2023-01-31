import re
from unidecode import unidecode
import csv
from enum import Enum
import pandas as panda
from dates import months, date_f1, date_f2,pattern_f
from random import sample
import os
import numpy as np

train_folder_path = "../train_folder_predilex/train_folder/txt_files/"
train_files_ids_path = "../train_folder_predilex/train_folder/x_train_ids.csv"

# Some french stopwords brought from Kaggle
stopwords = str = open('../stop_words_french.txt', 'r').read().split()

class Color(Enum):
    ACCIDENT = 1
    CONSOLIDATION = 2
    NONE = 3

def month_to_num(month):
    if month[0:3]=="jan":
        return 1
    elif month[0:3]=="fev":
        return 2
    elif month[0:3]=="mar":
        return 3
    elif month[0:3]=="avr":
        return 4
    elif month[0:3]=="mai":
        return 5
    elif month[0:4]=="juin":
        return 6
    elif month[0:4]=="juil":
        return 7
    elif month[0:3]=="aou":
        return 8
    elif month[0:3]=="sep":
        return 9
    elif month[0:3]=="oct":
        return 10
    elif month[0:3]=="nov":
        return 11
    elif month[0:3]=="dec":
        return 12
    else :
        return -1

# Loading the Y_train file
def get_dates():
    with open("../Y_train_predilex.csv", 'r') as file:
        y_data_rawtext = file.read()
    dates = re.split("\n\d+,.*?,",y_data_rawtext)
    dates.pop(0) #discard header
    list_of_dates =[]
    for i in range(len(dates)):
        accident,consolidation = dates[i].split(",")
        if len(accident)>4:
            yyyy,mm,dd = accident.split("-")
            acc = (int(dd),int(mm),int(yyyy))
        else:
            acc = ()
        if len(consolidation) > 4:
            yyyy, mm, dd = consolidation.split("-")
            cons = (int(dd), int(mm), int(yyyy))
        else:
            cons = ()
        list_of_dates.append([acc,cons])
    return list_of_dates

def clean_sentence(str):
    """The function is used to clean the sentence (text preprocessing)

    Args:
        str (string): a sentence

    Returns:
        string: the sentence cleaned up
    """
    # Cleaning the sentence :
    str = re.sub(r'[^\w\s]', ' ', str) # remove all punctuation and replace with space
    str = re.sub("\d+", "", str)       # remove all numerical characters
    str = re.sub(months,"",str)        # remove all months names
    str = re.sub(r'\b[a-zA-Z]{1,2}\b'," ", str) # remove all words that have a length equal to 2 or less
    str = re.sub(r"\b(%s)\b" % "|".join(stopwords), " ", str) # remove all stopwords in the list stopwords

    return str

def transform_dates_to_tuples(dates_format1,dates_format2):
    """
    Takes the dates in the two formats and converts them to a list tuples of int,
    this function standardizes the format of dates

    Args:
        dates_format1 (list): list of tuples of dates in first format (example : ('01','10','2000'))
        dates_format2 (list): list of tuples of dates in sencond format (example : ('1','octobre','2000'))

    Returns:
        list : a list of tuples of dates in this format : (1,10,2000)
    """
    list_of_dates = []
    if(len(dates_format1)!=0):  # there is at least one date in the line in the first format
        for i in range(0,len(dates_format1)):
            list_tmp=list(dates_format1[i])
            if(list_tmp[0]=="1er"):      # if the day is "1er" we assign 1 to it
                list_tmp[0]=1
            else:
                list_tmp[0]=int(dates_format1[i][0])
            list_tmp[1]=int(dates_format1[i][1])
            list_tmp[2]=int(dates_format1[i][2].replace(" ", ""))
            dates_format1[i]=tuple(list_tmp)
            
        list_of_dates.extend(dates_format1)
        
    if(len(dates_format2)!=0): # there is at least one date in the line in the second format
        for i in range(0,len(dates_format2)): # in the loop we change the months from str to int
            list_tmp=list(dates_format2[i])
            if(list_tmp[0]=="1er"):      # if the day is "1er" we assign 1 to it
                list_tmp[0]=1
            else:
                list_tmp[0]=int(dates_format2[i][0])
                
            list_tmp[1]=int(month_to_num(dates_format2[i][1]))
            list_tmp[2]=int(dates_format2[i][2].replace(" ", ""))
            
            dates_format2[i]=tuple(list_tmp)
            
        list_of_dates.extend(dates_format2)
    
    return list_of_dates

def labelize_sentences(dates_in_sentence,file_dates,line):
    """ This function is used to labelize the sub sentences of the according to the
    dates that occured in each part of the line.
    
    If the line contains more than one date sub sentences are created
    
    Example: FIRST_PART_OF_THE_SENTENCE date1 SECOND_PART_OF_THE_SENTENCE date2 THIRD_PART_OF_THE_SENTENCE
    
    FIRST_PART_OF_THE_SENTENCE + SECOND_PART_OF_THE_SENTENCE is the first sub sentence and the date that
    occurred in it is date1
    
    SECOND_PART_OF_THE_SENTENCE + THIRD_PART_OF_THE_SENTENCE is the second sub sentence and the date that
    occurred in it is date2
    

    Args:
        dates_in_sentence (list): the list of tuples corresponding
        to the dates that occured in the sentence
        
        file_dates (list): list of tuples corresponding the date
        of crime and date of consolidation of the current file
        
        line (string): the sentence

    Returns:
        list: a 2xn list of the sub_sentences and their labels
    """
    sentences_labelized = []  # the list of sub sentences that the function will return 
    labels = [] # the corresponding labels of these sub sentences
    isADate = False
    sub_sentences = [] # list of sentences where we reassemble the left and right part of the sentence
    sentence_split_on_date = re.split(pattern_f, line) # a list of sentence split whenever a date is found
    
    for i in range(0, len(dates_in_sentence)):
        left_part = clean_sentence(sentence_split_on_date[i])
        right_part = clean_sentence(sentence_split_on_date[i+1])
        sub_sentences.append(left_part+" "+right_part)
    
    pop_indexes = []
    
                                    
    for i  in range(0,len(dates_in_sentence)):
        
        if len(sub_sentences[i].split()) == 0:
            pop_indexes.append(i)
            continue
        
        isADate = False
        
        if dates_in_sentence[i] == file_dates[0]:
            sentences_labelized.append(sub_sentences[i].split())
            labels.append(0)
            isADate = True
            
        if dates_in_sentence[i] == file_dates[1] and isADate==False:
            sentences_labelized.append(sub_sentences[i].split())
            labels.append(1)
            isADate = True
            
        if isADate == False:
            sentences_labelized.append(sub_sentences[i].split())
            labels.append(2)
        
    for index in sorted(pop_indexes, reverse=True):
        del dates_in_sentence[index]
    
    # TODO : Delete this  
    """ 
    print("Sentence : ",line,"Sub sentences : ")
    for i  in range(0,len(dates_in_sentence)):
        print("     ",end=" ")
        print("Date : ",dates_in_sentence[i],end = "     ")
        print(sentences_labelized[i])
    
    print("Sentences_labelized",labels,"      :",sentences_labelized,"\n\n")
    """
    
    return sentences_labelized, labels, dates_in_sentence

def file_info(filename,file_dates):

    """
    file_info reads a file and returns the sentences where a date occured
    with the list of dates for each sentence.

    Args:
        filename (string): the name of the file
        file_dates (list): the date of accident and consolidation of that file

    Returns:
        list : a nx2 array where the sentences are in the first column and their
        corresponding labels are in the second column
    """
    file_path = train_folder_path + filename
    
    f = open(file_path, "r", encoding="utf8")
    
    labels = []
    total_sentences=[]
    total_dates=[]
    for line in f: # iterate over the lines of the text
        if(f!='\n'): # ignore blank lines
            line = unidecode(line) # remove accents and non-ascii characters
            line=line.lower()      # remove all capital letters
            dates  = re.findall(date_f1,line) # find the dates in the first format in the line
            dates2 = re.findall(date_f2,line) # find the dates in the second format in the line
            
            dates_in_sentences = transform_dates_to_tuples(dates, dates2)
            
            
            if(len(dates_in_sentences)!=0): # we found a least one date in the sentence
                
                sentences_labelized, label_of_sentence,dates_in_sentences = labelize_sentences(dates_in_sentences,file_dates,line)
                total_sentences.extend(sentences_labelized)
                labels.extend(label_of_sentence)
                total_dates.extend(dates_in_sentences)
                            
    return total_sentences,labels, total_dates

IDS_771 = range(771)

def create_dataframe(list_of_ids):
    all_sentences = []
    all_labels = []
    all_dates = []
    with open(train_files_ids_path, 'r', encoding="utf8") as file:
        text_dates = get_dates()
        csvreader = csv.reader(file)
        next(csvreader)                 # skip the first row [ID, filename]

        for row in csvreader:
            if(int(row[0]) not in list_of_ids):
                continue
            """ print('='*42,end=" ID : ")
            print(row[0],end=" ")
            print(row[1],end=" ")
            print('='*42) """
            file_sentences,file_labels, dates=file_info(row[1],text_dates[int(row[0])]) # row[0] is ID and row[1] is filename
            
            all_sentences.extend(file_sentences)
            all_labels.extend(file_labels)
            all_dates.extend(dates)
            
            """ for i in range(0,len(file_sentences)):
                print(file_sentences[i] , "\n    LABEL :", file_labels[i],end="\n\n")
            print('='*100) """
    d = {'Sentences': all_sentences, 'Label': all_labels,'Dates' : all_dates}

    df = panda.DataFrame(data=d)
    return df

Train_IDS = sample(IDS_771,int((70/100)*771))
# Test_IDS = IDS_771 - Train_IDS
Test_IDS = [i for i in IDS_771 if i not in Train_IDS]

df_train = create_dataframe(Train_IDS)
df_test = create_dataframe(Test_IDS)

df_0_1=df_train.loc[df_train['Label'].isin([0,1])]
df_2=df_train.loc[df_train['Label'] == 2]
df_2 = df_train.sample(n=int(len(df_0_1)))


#print(df_0_1.sort_values(by=['Label']).to_string())
#print(df_2.to_string())

frames = [df_0_1, df_2]
result = panda.concat(frames, ignore_index=True, sort=False)
df_train = result

# print(df_train.sort_values(by=['Label']).to_string())