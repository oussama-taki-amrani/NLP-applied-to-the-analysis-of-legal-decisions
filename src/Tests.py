# Tests
from Data_frame import file_info
from SVM import vec, classifier, encode_sentence, df_test, docs_tests_dates, docs_starts_test,sent_dates
import numpy as np

test_folder_path = "../test_folder_predilex/test_folder/txt_files/"


def get_dates2(sentences, dates, word_vec, clf):
    """
    This function takes a document, a classifier and a vector of words and output the predicted dates of accident and consolidation
    """
    x = np.zeros([len(sentences), len(vec)])
    for i in range(len(sentences)):
        x[i, :] = encode_sentence(sentences[i], word_vec)
    results = clf.decision_function(x)
    acc = np.argmax(results[:, 0])
    cons = np.argmax(results[:, 1])
    if results[acc, 0] > 0:
        date_accident = dates[acc]
    else:
        date_accident = ()
    if results[cons, 1] > 0:
        date_consolidation = dates[cons]
    else:
        date_consolidation = ()
    return [date_accident, date_consolidation]


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
    return get_dates2(sentences,dates,word_vec,clf)


test_sentences = np.array(df_test['Sentences'])
# print(get_dates("Agen_400518.txt",vec,classifier))

prec_per_date = 0
prec_per_doc = 0
for i in range(len(docs_tests_dates)-1):
    if (len(test_sentences[docs_starts_test[i]:docs_starts_test[i+1]]))==0:continue
    prediction = get_dates2(test_sentences[docs_starts_test[i]:docs_starts_test[i+1]],sent_dates[docs_starts_test[i]:docs_starts_test[i+1]],vec,classifier)
    print("Reelles  :", docs_tests_dates[i+1])
    print("Predites :", prediction)
    print("====================================================")
    if prediction==docs_tests_dates[i+1]: prec_per_doc+=1
    if (prediction[0] == docs_tests_dates[i+1][0]) : prec_per_date+=1
    if (prediction[1] == docs_tests_dates[i+1][1]) : prec_per_date+=1
prec_per_doc=100*prec_per_doc/(len(docs_tests_dates)-1)
prec_per_date = 100*prec_per_date/(2*(len(docs_tests_dates)-1))
print("Precision par document = ",prec_per_doc,"%")
print("Precision par date = ",prec_per_date,"%")