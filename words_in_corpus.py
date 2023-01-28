import csv
import re
import matplotlib.pyplot as plt

train_folder_path = "train_folder_predilex/train_folder/txt_files/"
train_files_ids_path = "train_folder_predilex/train_folder/x_train_ids.csv"

counts = dict()


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
def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i, y[i], y[i], ha = 'center',fontsize=7,position = (i,y[i]+1000))


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
        
    
counts = dict(sorted(counts.items(), key=lambda item: item[1],reverse = True))

Total = total_number_of_words(counts)

# Sorted list by length of words (to check if we will remove all words)
# That have a size of 2

lst2 = sorted(counts.keys(), key=len)
for elem in lst2:
    print(elem)

# A barplot of the most present words in the corpus

first_30_words = [[],[]]

for index, (key, value) in enumerate(counts.items()):
    if(index==30):
        break
    first_30_words[0].append(key)
    first_30_words[1].append(value)
    
plt.xticks(range(len(first_30_words[1])), first_30_words[0],rotation = 60,fontsize = 9)
  
plt.xlabel('Word')
plt.ylabel('Occurrences')

plt.title('Number of occurrences of each word , the total number of words in the corpus is 2 471 961')

addlabels(first_30_words[0], first_30_words[1])

plt.bar(range(len(first_30_words[1])), first_30_words[1]) 

plt.show()
