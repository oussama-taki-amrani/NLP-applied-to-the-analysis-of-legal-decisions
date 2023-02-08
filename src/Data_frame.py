import re
from unidecode import unidecode
import csv
from enum import Enum
import pandas as panda
from dates import months, date_f1, date_f2,pattern_f
from random import sample
from math import floor
from sklearn.model_selection import KFold
import numpy as np
import os

train_folder_path = "../train_folder_predilex/train_folder/txt_files/"
train_files_ids_path = "../train_folder_predilex/train_folder/x_train_ids.csv"

# Some french stopwords brought from Kaggle
stopwords = str = open('../stop_words_french.txt', 'r').read().split()

class Color(Enum):
    ACCIDENT = 1
    CONSOLIDATION = 2
    NONE = 3

def month_to_num(month):
    """Converts a month to a number

    Args:
        month (Str): The month to convert

    Returns:
        int: it corresponding number
    """
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
    #str = re.sub(r"\b(%s)\b" % "|".join(stopwords), " ", str) # remove all stopwords in the list stopwords

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
            if int(list_tmp[2])<1000:
                if int(list_tmp[2])<23:
                    list_tmp[2] = list_tmp[2] + 2000
                else :
                    list_tmp[2] = list_tmp[2] + 1000
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

def label_sentences(dates_in_sentence,file_dates,line):
    """ This function is used to label the sub sentences of the according to the
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
    if len(dates_in_sentence)==0:
        return [],[],dates_in_sentence
    sentences_labeled = []  # the list of sub sentences that the function will return 
    labels = [] # the corresponding labels of these sub sentences
    isADate = False
    sub_sentences = [] # list of sentences where we reassemble the left and right part of the sentence
    sentence_split_on_date = re.split(pattern_f, line) # a list of sentence split whenever a date is found
    sentence_split_on_date.append(" ") # in case the last part of the sentence is a date
    W = 10 # Half size of the windows
    
    
    for i in range(0, len(dates_in_sentence)):
        left_part = clean_sentence(sentence_split_on_date[i])
        right_part = clean_sentence(sentence_split_on_date[i+1])
        right = right_part.split() # right part of the date converted to list of words
        left = left_part.split() # left part of the date converted to list of words
        len_l = len(left)
        len_r = len(right)
        
        # If one part has more words than the other one, we will take more words from
        # the long part and add them to the list, until there are no more words or
        # that we reached the max length == 2*W
        if len_l < W or len_r < W :
            
            if(len(right_part.split())>=W):
                right_part = [right[indx] for indx in range(0,min(len_r,2*W-len_l))]
            else:
                right_part = right
            
            if(len(left)>=W):
                left_part = [left[indx] for indx in range(0,min(2*W-len_r,len_l),len_l)]
            else:
                left_part = left
        else:
            right_part = right[0:W]
            left_part = left[len_l-W:len_l]
            
        sub_sentences.append(left_part+right_part)
    pop_indexes = []
    
    # In the following loop the lists of words are labeled                                    
    for i  in range(0,len(dates_in_sentence)):
        
        if len(sub_sentences[i]) == 0:
            pop_indexes.append(i)
            continue
        
        isADate = False
        
        if dates_in_sentence[i] == file_dates[0]:
            sentences_labeled.append(sub_sentences[i])
            labels.append(0)
            isADate = True
            
        if dates_in_sentence[i] == file_dates[1] and isADate==False:
            sentences_labeled.append(sub_sentences[i])
            labels.append(1)
            isADate = True
            
        if isADate == False:
            sentences_labeled.append(sub_sentences[i])
            labels.append(2)
        
    for index in sorted(pop_indexes, reverse=True):
        del dates_in_sentence[index]
    
    return sentences_labeled, labels, dates_in_sentence

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
                
                sentences_labeled, label_of_sentence,dates_in_sentences = label_sentences(dates_in_sentences,file_dates,line)
                total_sentences.extend(sentences_labeled)
                labels.extend(label_of_sentence)
                total_dates.extend(dates_in_sentences)
                            
    return total_sentences,labels, total_dates

IDS_771 = range(771)

def create_dataframe(list_of_ids):
    """This function creates a dataframe corresponding to the documents in the list of ids
    given in the list as a parameter
    
    It iterates over the lines of each document, extracts the sentences where a date was
    found.
    
    For each date found, we take a window of fixed size : W , of words surrounding the date.
    
    We label this list of words with one of the three labels 0, 1 or 2, depending on
    whether the date is respectively an accident date, a consolidation date or none of these.
    

    Args:
        list_of_ids (list): a list of IDs of the training dataset

    Returns:
        panda dataframe: A dataframe where all the list of the words where labeled
    """
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
            
            file_sentences,file_labels, dates=file_info(row[1],text_dates[int(row[0])]) # row[0] is ID and row[1] is filename
            # We gather all the similar dates together by concataneting their lists :
            temp_data = [[],[],[]]
            for i in range(len(dates)):
                if dates[i] not in temp_data[0]:
                    temp_data[0].append(dates[i])
                    temp_data[1].append(file_sentences[i])
                    temp_data[2].append(file_labels[i])
                else:
                    idx = temp_data[0].index(dates[i])
                    temp_data[1][idx].extend(file_sentences[i])
            
            all_sentences.extend(file_sentences)
            all_labels.extend(file_labels)
            all_dates.extend(dates)
            
    d = {'Sentences': all_sentences, 'Label': all_labels,'Dates' : all_dates}

    df = panda.DataFrame(data=d)
    return df

def select_80_percent():
    IDs = [0]
    curr_city = ''
    with open(train_files_ids_path, 'r', encoding="utf8") as file:
        csvreader = csv.reader(file)
        next(csvreader)                  # skip the first row [ID, filename]
        row =  next(csvreader)
        curr_city = row[1][0:3]
        for index, row in enumerate(csvreader):
            if row[1][0:3] != curr_city:
                curr_city = row[1][0:3]
                IDs.append(index-2)
    IDs.append(771)
    return IDs  
                
IDs = select_80_percent()

"""
def split_datas(K,ids):
    
    # K : number of data chunks
    # ids : list of ids to split
    
    ids = np.array(ids)
    k_datas = []
    step = len(ids)//K
    start = 0
    end = step
    for k in range(K):
        if k==K-1:
            k_datas.append(ids[start:])
            continue
        k_datas.append(ids[start:end])
        start += step
        end += step
    return k_datas
"""        


K = 5
# splited_IDs = split_datas(K,IDS_771)

kf = KFold(n_splits=K,shuffle=True)
kf.get_n_splits(IDS_771)
"""
for i, (Train_IDS, Test_IDS) in enumerate(kf.split(IDS_771)):
    print(f"  Train: {Train_IDS}")
    print(f"  Test:  {Test_IDS}")
    
    df_train = create_dataframe(Train_IDS)
    df_test = create_dataframe(Test_IDS)
    
    df_0_1=df_train.loc[df_train['Label'].isin([0,1])]
    df_2=df_train.loc[df_train['Label'] == 2]
    
    frames = [df_0_1, df_2]
    result = panda.concat(frames, ignore_index=True, sort=False)
    df_train = result
"""

"""
for i in range(0, len(IDs)-1):
    Train_IDS.extend(sample(range(IDs[i], IDs[i+1]), floor((80/100)*(IDs[i+1] - IDs[i]))))



Train_IDS.sort()

Test_IDS = [i for i in IDS_771 if i not in Train_IDS]

df_train = create_dataframe(Train_IDS)
df_test = create_dataframe(Test_IDS)

df_0_1=df_train.loc[df_train['Label'].isin([0,1])]
df_2=df_train.loc[df_train['Label'] == 2]
#df_2 = df_train.sample(n=floor(len(df_0_1)))

frames = [df_0_1, df_2]
result = panda.concat(frames, ignore_index=True, sort=False)
df_train = result

"""