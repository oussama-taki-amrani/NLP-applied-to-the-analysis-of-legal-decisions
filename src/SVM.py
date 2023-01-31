from Data_frame import df_train
import numpy as np
from sklearn import svm, linear_model, ensemble
import json


def compute_words_occurence(labelized_sentences):

    occurences = [{},{},{}]
    for i in range(len(labelized_sentences['Label'])):
        sentence = labelized_sentences['Sentences'][i]
        label = labelized_sentences['Label'][i]
        for word in sentence:
            if word not in occurences[0]:
                occurences[0][word] = 0
                occurences[1][word] = 0
                occurences[2][word] = 0
            occurences[label][word] += 1
    return occurences

def scores_from_occurences(occurences,alpha=1):
    scores = [{},{},{}]
    for word in occurences[0]:
        scores[0][word] = occurences[0][word] / (1 + alpha*(occurences[1][word] + occurences[2][word]))
        scores[1][word] = occurences[1][word] / (1 + alpha*(occurences[0][word] + occurences[2][word]))
        scores[2][word] = occurences[2][word] / (1 + alpha*(occurences[0][word] + occurences[1][word]))
    return scores

#Sorting scores
def get_best_scores(scores,nb=50):
    sorted_scores = [{}, {}, {}]
    sorted_scores[0] =  dict(sorted(scores[0].items(), key=lambda x: x[1],reverse=True))
    sorted_scores[1] =  dict(sorted(scores[1].items(), key=lambda x: x[1],reverse=True))
    sorted_scores[2] =  dict(sorted(scores[2].items(), key=lambda x: x[1],reverse=True))
    vec = list(sorted_scores[0].keys())[0:nb] + list(sorted_scores[1].keys())[0:nb] + list(sorted_scores[2].keys())[0:nb]
    return vec,sorted_scores



#Bag of words
def encode_sentence(sentence,words_vec):
    x = []
    for j in range(len(words_vec)):
        x.append(sentence.count(words_vec[j]))
    return x



# all_sentences = np.array(df['Sentences'])
# all_labels = np.array(df['Label'])
# df_train = {'Sentences':all_sentences[:docs_starts[nb_docs_train]],'Label':all_labels[:docs_starts[nb_docs_train]]}
# df_test = {'Sentences':all_sentences[docs_starts[nb_docs_train]:],'Label':all_labels[docs_starts[nb_docs_train]:]}
# docs_starts_test = np.array(docs_starts)[nb_docs_train:] - docs_starts[nb_docs_train]
# print("DOCS:", docs_starts_test[:10])
# print("S:", len(text_dates)+1,"P: ",nb_docs)
# docs_tests_dates = [text_dates[i] for i in range(nb_docs_train-1,nb_docs-1)]
# sent_dates = []
# for i in range(docs_starts[nb_docs_train],len(all_sentences)):
#     sent_dates.append(all_dates[i])



#Computing scores
occurences = compute_words_occurence(df_train)
vec, sorted_scores = get_best_scores(scores_from_occurences(occurences))
with open("../outputs/sorted_scores_accident.txt", 'w') as file:
    file.write(json.dumps(sorted_scores[0]))
with open("../outputs/sorted_scores_consolidation.txt", 'w') as file:
    file.write(json.dumps(sorted_scores[1]))
with open("../outputs/sorted_scores_other.txt", 'w') as file:
    file.write(json.dumps(sorted_scores[2]))



# Setting up the training datas
x_train = np.zeros([len(df_train['Sentences']), len(vec)])
y_train = np.zeros(len(df_train['Label']))
print(df_train['Label'])
for i in range(len(df_train['Label'])):
    x_train[i,:] = encode_sentence(df_train['Sentences'][i],vec)
    y_train[i] = df_train['Label'][i]
#Training the SVM
# classifier = svm.SVC(kernel='linear')
classifier = linear_model.LogisticRegression()
# classifier = ensemble.RandomForestClassifier()
print("Start learning")
print("taille",len(y_train))
classifier.fit(x_train,y_train)



