import library as ge
import pandas as pd


# open the test files in the folder
files = open("train_folder_predilex\\train_folder\\x_train_ids.csv", "r",encoding="utf8")
# each line is a file name with number id with uft8 encoding
list_files = files.readlines()
# eliminate until the first comma
list_files = [line.split(",")[1] for line in list_files]
# eliminate the \n
list_files = [line.strip() for line in list_files]
# eliminate the first line
list_files.pop(0)
for i in range(0, len(list_files)):
    list_files[i] ="train_folder_predilex\\train_folder\\txt_files\\"+ list_files[i] 


df = ge.fill_df(list_files)
# print the textId




labels = ge.label_extraction()
# split the data frame in train and test from textID 0 to 20

df_train = df[df["textId"] <= 20]
df_test = df[df["textId"] > 20]


# fill the labels 
for i in range(len(labels)):
    df_train["labels"][i] = labels[i]





# print the list of textId unique



# train the model
model = ge.train_model(df_train)

# predict the labels
df_test = ge.predict_labels(df_test, model)


# predictedSex is a list of predictedSex
predictedSex = ge.predict_sex(df_test)


# open csv file
csv_file = open("Y_train_predilex.csv", "r", encoding="utf-8")
# read the file
lines = csv_file.readlines()
# delete the first line
lines = lines[1:]
# for each line take the second element after the comma
lines = [line.split(",")[1] for line in lines]
print(len(lines))

for i in range(len(predictedSex)):
    print(predictedSex[i]," and the correct one is :",lines[i])

print("Accuracy: ", ge.precision(predictedSex, lines))
tp, tn, fp, fn = ge.metrics(predictedSex, lines)
print("true positives: ", tp)
print("true negatives: ", tn)
print("false positives: ", fp)
print("false negatives: ", fn)

