# Tests
from Data_frame import file_info
from SVM import vec, classifier, encode_sentence
import numpy as np

test_folder_path = "../test_folder_predilex/test_folder/txt_files/"


def get_dates(file_name, word_vec, clf):
    """
    This function takes a document, a classifier and a vector of words and output the predicted dates of accident and consolidation
    """
    # Do this when the file is in the test folder:
    file_name = "../../"+ test_folder_path + file_name
    # The true dates of the the file are supposed unknown
    sentences,l,dates = file_info(file_name, (00, 00, 0000))
    print("DATES")
    print(dates)
    x = np.zeros([len(sentences),len(vec)])
    for i in range(len(sentences)):
        x[i,:] = encode_sentence(sentences[i],word_vec)
    results = clf.decision_function(x)
    print(results.shape)
    print(len(dates))
    print(len(sentences))
    acc = np.argmax(results[:,0])
    cons = np.argmax(results[:,1])
    if results[acc,0]>0:
        date_accident = dates[acc]
    else:
        date_accident = "n.a."
    if results[cons,1]>0:
        date_consolidation = dates[cons]
    else:
        date_consolidation = "n.c."
    print(sentences[acc])
    print(sentences[cons])
    return date_accident,date_consolidation


print(get_dates("Aix-en-provence_619114.txt",vec,classifier))