import csv
from Data_frame import get_dates, file_info
import numpy as np
import matplotlib.pyplot as plt

def unique(list1):
 
    # initialize a null list
    unique_list = []
 
    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
            
    return len(unique_list)

# function to add value labels (from geeksforgeeks)
def addlabels(x,y,decalage):
    for i in range(len(x)):
        plt.text(i, y[i], str(y[i])+" %", ha = 'center',fontsize=12,position = (i,y[i] + decalage))

train_folder_path = "../train_folder_predilex/train_folder/txt_files/"
train_files_ids_path = "../train_folder_predilex/train_folder/x_train_ids.csv"

total_dates_in_corpus = 0
min = 1000
max = 0
text_indx = 0

total_number_of_dates_of_accident = 0
min_acc = 1000
max_acc = 0

total_number_of_dates_of_cons = 0
min_cons = 1000
max_cons = 0

unique_dates_in_corpus = 0

with open(train_files_ids_path, 'r', encoding="utf8") as file:
    text_dates = get_dates()
    csvreader = csv.reader(file)
    next(csvreader)                 # skip the first row [ID, filename]
    
    for i, row in enumerate(csvreader):
        file_sentences,file_labels, dates=file_info(row[1],text_dates[int(row[0])]) # row[0] is ID and row[1] is filename
        
        total_number_of_dates_of_accident += file_labels.count(0)
        total_number_of_dates_of_cons += file_labels.count(1)
        
        if file_labels.count(0)<min_acc:
            min_acc = file_labels.count(0)
        if file_labels.count(0)>max_acc:
            max_acc = file_labels.count(0)
        
        if file_labels.count(1)<min_cons:
            min_cons = file_labels.count(1)
        if file_labels.count(1)>max_cons:
            max_cons = file_labels.count(1)
        
        total_dates_in_corpus += len(dates)
        if len(dates)<min:
            min = len(dates)
        if len(dates)>max:
            max = len(dates)
            text_indx = i
            
        unique_dates_in_corpus += unique(dates)

print("The total number of dates in the corpus is : ",total_dates_in_corpus)
print("With an average number of dates per text of: ",total_dates_in_corpus/770)
print("The maximum number of dates per text is: ", max)
print("The minimum number of dates per text is: ", min)


print("\nThe total number of accident dates in the corpus is : ",total_number_of_dates_of_accident)
print("With an average number of dates per text of: ", total_number_of_dates_of_accident/770)
print("The maximum number of dates of accident per text is: ", max_acc)
print("The minimum number of dates of accident per text is: ", min_acc)

print("\nThe total number of consolidation dates in the corpus is : ",total_number_of_dates_of_cons)
print("With an average number of dates per text of: ", total_number_of_dates_of_cons/770)
print("The maximum number of dates of consolidation per text is: ", max_cons)
print("The minimum number of dates of consolidation per text is: ", min_cons)

print("\nThe number of unique dates per text is: ", unique_dates_in_corpus/770)

# Make a random dataset:
height = [0.1253*100, 0.0304*100, 0.8441*100]
bars = ('Accident', 'Consolidation', 'None')

y_pos = np.arange(len(bars))

# Create bars
plt.bar(y_pos, height, color = ['green', 'red', 'blue'])

plt.title('Percentage of the training data of each class')

# Create names on the x-axis
plt.xticks(y_pos, bars)

addlabels(bars,height,1)
# Show graphic
plt.show()