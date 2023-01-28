import re
from unidecode import unidecode
import csv

train_folder_path = "train_folder_predilex/train_folder/txt_files/"
train_files_ids_path = "train_folder_predilex/train_folder/x_train_ids.csv"

stopwords = str = open('stop_words_french.txt', 'r').read().split()

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


def file_info(filename):
    """
    file_info reads a file and returns the sentences where a date occured
    with the list of dates for each sentence.
    
    :param filename: the name of the file
    
    :return: a 2xn array where the sentences are in the first line and their
    corresponding dates are in the second line
    """
    file_path = train_folder_path + filename
    
    f = open(file_path, "r")

    total_sentences=[[],[]]

    for line in f: # iterate over the lines of the text
        if(f!='\n'): # ignore blank lines
            line = unidecode(line) # remove accents and non-ascii characters
            line=line.lower()      # remove all capital letters
            dates  = re.findall(date_f1,line) # find the dates in the first format in the line
            dates2 = re.findall(date_f2,line) # find the dates in the second format in the line
            temp = []
            if(len(dates)!=0):  # there is at least one date in the line in the first format
                temp.extend(dates)
            if(len(dates2)!=0): # there is at least one date in the line in the second format
                for i in range(0,len(dates2)): # in the loop we change the months from str to int
                    list_tmp=list(dates2[i])
                    list_tmp[1]=month_to_num(dates2[i][1])
                    if(list_tmp[0]=="1er"):      # if the day is "1er" we assign 1 to it
                        list_tmp[0]=1
                    dates2[i]=tuple(list_tmp)
                temp.extend(dates2)
            if(len(temp)!=0): # we found a least one date in the sentence
                # Cleaning the sentence :
                line = re.sub(r'[^\w\s]', ' ', line) # remove all punctuation and replace with space
                line = re.sub("\d+", "", line)       # remove all numerical characters
                line = re.sub(months,"",line)        # remove all months names
                line = re.sub(r'\b[a-zA-Z]{1,2}\b'," ", line) # remove all words that have a length equal to 1 or less
                line = re.sub(r"\b(%s)\b" % "|".join(stopwords), " ", line) # remove all stopwords in the list stopwords
                
                if(len(line.split())!=0):   # There are still words in the sentence after cleaning it
                    total_sentences[0].append(line.split())
                    total_sentences[1].append(temp)
    return total_sentences    
    
    
# These are the words that occur many times in the corpus but give 0 meaning :
non_meaningfull_words = ""

# ------- Date formats --------------------

dd = "(0[1-9]| [1-9]|1[0-9]|2[0-9]|3[0-1]|1er)"
mm = "(0[1-9]|1[0-2])"
yyyy = "([1-2] {0,1}[0-9][0-9][0-9])" # allows max one space in y yyy

# date in format dd/ or dd- or dd.  with dd between 01 and 31 :
dd_f1 = dd + " *[(\/|\-|\.)]"

# month in format mm/ or mm- or mm.  with mm between 01 and 12 :
mm_f1 = " *" + mm + " *[(\/|\-|\.)]"

# year in format yyyy  with yyyy between 1000 and 2999 :
yy_f1 = " *" + yyyy

# date in the dd/mm/yyyy or dd-mm-yyyy or dd.mm.yyyy format (spaces allowed between / and numbers)
date_f1 = dd_f1 + mm_f1 + yy_f1

months = "(janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre|decembre|fevrier|aout)"

# date in the dd month yyyy format (spaces allowed between / and numbers)
date_f2 = dd + " *" + months + " *" + yyyy

# date format :

date_f = "(" + date_f1 + "|" + date_f2 + ")"

# -----------------------------------------



# ----------------------- Testing the output for all files --------------------------------

with open(train_files_ids_path, 'r') as file:
    csvreader = csv.reader(file)
    next(csvreader)                 # skip the first row [ID, filename]
    
    for row in csvreader:
        print('='*42,end=" ")
        print(row[1],end=" ")
        print('='*42)
        file_sentences=file_info(row[1])
        
        for i in range(0,len(file_sentences[0])):
            print(file_sentences[0][i] , "\n    DATES :", file_sentences[1][i],end="\n\n")
        print('='*100)


# ---------------------- Testing with this file ------------------------------------------------------
"""

total_sentences = file_info("Bastia_1300101.txt")


for i in range(0,len(total_sentences[0])):
    print(total_sentences[0][i] , "\n    DATES :", total_sentences[1][i],end="\n\n")
"""
# ---------------------------------------------------------------------------------------------------