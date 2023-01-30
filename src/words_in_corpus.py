import csv
import re
import matplotlib.pyplot as plt
from Data_frame import df

train_folder_path = "../train_folder_predilex/train_folder/txt_files/"
train_files_ids_path = "../train_folder_predilex/train_folder/x_train_ids.csv"

counts = dict()
counts_after_removing_stop_words = dict()


def word_count(counts,str):
    """
    This function reads in the text and increments the number of occurrences
    of each word whenever encountered.
    In case the word is not found in the dictionary it is created

    Args:
        counts (_dict_): A dictionary of words' counts
        str (_string_): The text  

    Returns:
        _dict_ : The original dictionary modified
    """
    words = str.split()

    for word in words:
        if word in counts:
            counts[word] += 1
        else:
            counts[word] = 1

    return counts

def total_number_of_words(counts):
    TotalNumberOfWords = 0

    for word,count in counts.items() :
        TotalNumberOfWords = TotalNumberOfWords + count
        print(word,end=" : ")
        print(count)
    print(TotalNumberOfWords)
    return TotalNumberOfWords # TotalNumberOfWords = 2 471 961



# function to add value labels (from geeksforgeeks)
def addlabels(x,y,decalage):
    for i in range(len(x)):
        plt.text(i, y[i], y[i], ha = 'center',fontsize=7,position = (i,y[i] + decalage))

def plot_dict(dictionnary, decalage):
    # A barplot of the most 30 present words in the corpus after cleaning them

    first_30_words = [[],[]]

    for index, (key, value) in enumerate(dictionnary.items()):
        if(index==30):
            break
        first_30_words[0].append(key)
        first_30_words[1].append(value)
        
    plt.xticks(range(len(first_30_words[1])), first_30_words[0],rotation = 60,fontsize = 7)
    
    plt.xlabel('Word')
    plt.ylabel('Occurrences')

    n = '{:,}'.format(total_number_of_words(dictionnary)).replace(',',' ')
    plt.title('Number of occurrences of each word , the total number of words in the corpus is : %s' % n )

    addlabels(first_30_words[0], first_30_words[1], decalage)

    plt.bar(range(len(first_30_words[1])), first_30_words[1]) 

    plt.show()

with open(train_files_ids_path, 'r') as file:
    csvreader = csv.reader(file)
    next(csvreader)
    
    for row in csvreader:
        filename = train_folder_path + row[1]
        f = open(filename, "r")
        content = f.read()
        content = re.sub(r'[^\w\s]', ' ', content) # remove all punctuation and replace with space
        content = re.sub("\d+", "", content)       # remove all numerical characters
        counts = word_count(counts,content)
        
# After removing all stop words
for row in df.loc[:,"Sentences"]:
    for elem in row:
        counts_after_removing_stop_words = word_count(counts_after_removing_stop_words,elem)
    
counts = dict(sorted(counts.items(), key=lambda item: item[1],reverse = True))
counts_after_removing_stop_words = dict(sorted(counts_after_removing_stop_words.items(), key=lambda item: item[1],reverse = True))

Total = total_number_of_words(counts)
Total2 = total_number_of_words(counts_after_removing_stop_words)

# Sorted list by length of words (to check if we will remove all words)
# That have a size of 2

"""
lst2 = sorted(counts.keys(), key=len)
for elem in lst2:
    print(elem)
"""

plot_dict(counts,1000)
plot_dict(counts_after_removing_stop_words,50)