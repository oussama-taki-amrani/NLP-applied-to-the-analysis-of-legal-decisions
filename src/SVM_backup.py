from Data_frame import df_train
import numpy as np
from sklearn import svm


def compute_words_occurence(labelized_sentences):

    occurences = [{},{},{}]
    for i in range(len(labelized_sentences)):
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


occurences = compute_words_occurence(df_train)
vec,sorted_scores = get_best_scores(scores_from_occurences(occurences))

for k,v in sorted_scores[0].items():
    print(k,(30-len(k))*" ",":",v,end="      ")
    print(occurences[0][k],"       ",occurences[1][k]+occurences[2][k])
for k,v in sorted_scores[1].items():
    print(k,(30-len(k))*" ",":",v,end="      ")
    print(occurences[1][k],"       ",occurences[0][k]+occurences[2][k])
for k,v in sorted_scores[2].items():
    print(k,(30-len(k))*" ",":",v,end="      ")
    print(occurences[2][k],"       ",occurences[1][k]+occurences[0][k])

#Bag of words
def encode_sentence(sentence,words_vec):
    x = []
    for j in range(len(words_vec)):
        x.append(sentence.count(words_vec[j]))
    return x


# Setting up the training datas
x_train = np.zeros([len(df_train),len(vec)])
y_train = np.zeros(len(df_train))
for i in range(len(df_train)):
    x_train[i,:] = encode_sentence(df_train['Sentences'][i],vec)
    y_train[i] = df_train['Label'][i]

#Training the SVM
classifier = svm.SVC(kernel='linear')
print("Start learning")
classifier.fit(x_train,y_train)



