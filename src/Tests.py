# Tests
from Data_frame import create_dataframe, Test_IDS, get_dates
from SVM import vec, classifier, encode_sentence
import numpy as np

class colors:
    reset = '\033[0m'
    class fg:
            red = '\033[31m'
            green = '\033[32m'
        

test_folder_path = "../test_folder_predilex/test_folder/txt_files/"


def predict_dates(sentences, dates, word_vec, clf):
    """
    This function takes a document, a classifier and a vector of words and output the predicted dates of accident and consolidation
    """
    x = np.zeros([len(sentences), len(vec)])
    for i in range(len(sentences)):
        x[i, :] = encode_sentence(sentences[i], word_vec)
    results = clf.predict_proba(x)
    acc = np.argmax(results[:, 0])
    cons = np.argmax(results[:, 1])
    lim = .2
    if results[acc, 0] > .1:
        date_accident = dates[acc]
    else:
        date_accident = ()
    if results[cons, 1] > lim:
        date_consolidation = dates[cons]
    else:
        date_consolidation = ()
    return [date_accident, date_consolidation]


def predict_dates_from_filename(file_name, word_vec, clf):
    """
    This function takes a document, a classifier and a vector of words and output the predicted dates of accident and consolidation
    """
    # Do this when the file is in the test folder:
    file_name = "../../"+ test_folder_path + file_name
    # The true dates of the the file are supposed unknown
    sentences,l,dates = create_dataframe(file_name, (00, 00, 0000))
    print("DATES")
    print(dates)
    return predict_dates(sentences,dates,word_vec,clf)


# test_sentences = np.array(df_test['Sentences'])
# print(get_dates("Agen_400518.txt",vec,classifier))







true_dates = get_dates()
prec_per_date = 0
prec_per_doc = 0
acc_per_classe = [0,0]
for i in range(len(Test_IDS)):
    doc_df = create_dataframe([Test_IDS[i]])
    print("ID  :", Test_IDS[i])
    sentences = doc_df['Sentences']
    if (len(sentences)) == 0:
        continue
    dates = doc_df['Dates']
    prediction = predict_dates(sentences,dates,vec,classifier)
    t_dates = true_dates[Test_IDS[i]]
    print("Reelles  :", t_dates)
    # print("Predites :", prediction)
    print("Predites :", end =" [")
    if prediction[0]==t_dates[0]:
        print (colors.fg.green, prediction[0], colors.reset, end = ", ")
    else:
        print (colors.fg.red, prediction[0], colors.reset , end = ", ")
    if prediction[1]==t_dates[1]:
        print (colors.fg.green, prediction[1], colors.reset, end = "]\n")
    else:
        print (colors.fg.red, prediction[1], colors.reset, end = "]\n")
    print(colors.reset)    
    print("====================================================")
    if prediction==t_dates: prec_per_doc+=1
    if (prediction[0] == t_dates[0]) : 
        prec_per_date+=1
        acc_per_classe[0] +=1
    if (prediction[1] == t_dates[1]) : 
        prec_per_date+=1
        acc_per_classe[1] +=1
prec_per_doc = 100*prec_per_doc/(len(Test_IDS))
acc_per_classe[0] = 100*acc_per_classe[0]/(len(Test_IDS))
acc_per_classe[1] = 100*acc_per_classe[1]/(len(Test_IDS))
prec_per_date = 100*prec_per_date/(2*(len(Test_IDS)))
print("Precision par document = ",prec_per_doc,"%")
print("Precision par date = ",prec_per_date,"%")
print("Precision accident = ",acc_per_classe[0],"%")
print("Precision consoldiation = ",acc_per_classe[1],"%")