import re
from unidecode import unidecode
import csv
from enum import Enum
import pandas as panda
from dates import months, date_f1, date_f2

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
    str = re.sub(r'\b[a-zA-Z]{1,2}\b'," ", str) # remove all words that have a length equal to 1 or less
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

def labelize_sentence(dates_in_sentences,file_dates,line):
    """ This function is used to labelize a sentence according to the
    date that occured in it.
    
    If the sentence contains both the date of accident and consolidation
    then it's duplicated and is given two labels

    Args:
        dates_in_sentences (list): the list of tuples corresponding
        to the dates that occured in the sentence
        
        file_dates (list): list of tuples corresponding the date
        of crime and date of consolidation of the current file
        
        line (string): the sentence

    Returns:
        list: a 2xn list of the sentence and its labels
    """
    sentence_labelized = []
    labels = []
    isADate = False
    for date in dates_in_sentences:
        if date == file_dates[0]:
            sentence_labelized.append(line.split())
            labels.append(0)
            isADate = True
        if date == file_dates[1]:
            sentence_labelized.append(line.split())
            labels.append(1)
            isADate = True
    if isADate == False:
        sentence_labelized.append(line.split())
        labels.append(2)
    
    return sentence_labelized, labels

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
    total_dates = []
    for line in f: # iterate over the lines of the text
        if(f!='\n'): # ignore blank lines
            line = unidecode(line) # remove accents and non-ascii characters
            line=line.lower()      # remove all capital letters
            dates  = re.findall(date_f1,line) # find the dates in the first format in the line
            dates2 = re.findall(date_f2,line) # find the dates in the second format in the line
            
            dates_in_sentences = transform_dates_to_tuples(dates, dates2)
            
            
            if(len(dates_in_sentences)!=0): # we found a least one date in the sentence
                line = clean_sentence(line)
                if(len(line.split())!=0):   # There are still words in the sentence after cleaning it
                    sentence_labelized, label_of_sentence = labelize_sentence(dates_in_sentences,file_dates,line)
                    total_sentences.extend(sentence_labelized)
                    labels.extend(label_of_sentence)
                    # if len(dates_in_sentences)>1 : print("Got ",len(dates_in_sentences)," conflicting dates")
                    #Just take the first date for now
                    total_dates.append(dates_in_sentences[0])
                            
    return total_sentences,labels,total_dates


# ----------------------- Output for all files --------------------------------
all_sentences = []
all_labels = []

with open(train_files_ids_path, 'r', encoding="utf8") as file:
    text_dates = get_dates()
    csvreader = csv.reader(file)
    next(csvreader)                 # skip the first row [ID, filename]
    
    for row in csvreader:
        """ print('='*42,end=" ID : ")
        print(row[0],end=" ")
        print(row[1],end=" ")
        print('='*42) """
        file_sentences,file_labels,dates=file_info(row[1],text_dates[int(row[0])]) # row[0] is ID and row[1] is filename
        
        all_sentences.extend(file_sentences)
        all_labels.extend(file_labels)
        
        """ for i in range(0,len(file_sentences)):
            print(file_sentences[i] , "\n    LABEL :", file_labels[i],end="\n\n")
        print('='*100) """

d = {'Sentences': all_sentences, 'Label': all_labels}

df = panda.DataFrame(data=d)

# print(df.to_string())
