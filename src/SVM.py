from Data_frame import create_dataframe
import numpy as np
from sklearn import svm, linear_model, ensemble
import json


def compute_words_occurence(labelized_sentences):

    occurences = [{},{},{}]
    total_of_words_per_class = [0,0,0]
    for i in range(len(labelized_sentences['Label'])):
        sentence = labelized_sentences['Sentences'][i]
        label = labelized_sentences['Label'][i]
        for word in sentence:
            if word not in occurences[0]:
                occurences[0][word] = 0
                occurences[1][word] = 0
                occurences[2][word] = 0
            occurences[label][word] += 1
            total_of_words_per_class[label]+=1
    total_of_words = sum(total_of_words_per_class)
    percentage_of_words = [x / total_of_words for x in total_of_words_per_class]
    # print(percentage_of_words)
    return occurences,percentage_of_words

def scores_from_occurences(occurences,percentage,alpha=0.2):
    scores = [{},{},{}]
    for word in occurences[0]:
        scores[0][word] = occurences[0][word] / (1 + alpha*(occurences[1][word] + 0.05*occurences[2][word]))
        scores[1][word] = occurences[1][word] / (1 + alpha*(occurences[0][word] + 0.05*occurences[2][word]))
        scores[2][word] = occurences[2][word] / (1 + alpha*(5*occurences[0][word] + 5*occurences[1][word]))
    return scores

#Sorting scores
def get_best_scores(scores,nb=[80,30,100]):
    sorted_scores = [{}, {}, {}]
    sorted_scores[0] =  dict(sorted(scores[0].items(), key=lambda x: x[1],reverse=True))
    sorted_scores[1] =  dict(sorted(scores[1].items(), key=lambda x: x[1],reverse=True))
    sorted_scores[2] =  dict(sorted(scores[2].items(), key=lambda x: x[1],reverse=True))
    vec = list(sorted_scores[0].keys())[0:nb[0]] + list(sorted_scores[1].keys())[0:nb[1]] + list(sorted_scores[2].keys())[0:nb[2]]
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
# occurences,percentages = compute_words_occurence(df_train)
# vec, sorted_scores = get_best_scores(scores_from_occurences(occurences,percentages))
# with open("../outputs/sorted_scores_accident.txt", 'w') as file:
#     file.write(json.dumps(sorted_scores[0]))
# with open("../outputs/sorted_scores_consolidation.txt", 'w') as file:
#     file.write(json.dumps(sorted_scores[1]))
# with open("../outputs/sorted_scores_other.txt", 'w') as file:
#     file.write(json.dumps(sorted_scores[2]))

# for k,v in sorted_scores[0].items():
#     print(k,(30-len(k))*" ",":",v,end="      ")
#     print(occurences[0][k],"       ",occurences[1][k]+occurences[2][k])
# print("="*100)
# for k,v in sorted_scores[1].items():
#     print(k,(30-len(k))*" ",":",v,end="      ")
#     print(occurences[1][k],"       ",occurences[0][k]+occurences[2][k])
# print("="*100)
# for k,v in sorted_scores[2].items():
#     print(k,(30-len(k))*" ",":",v,end="      ")
#     print(occurences[2][k],"       ",occurences[1][k]+occurences[0][k])
# print("="*100)

# Setting up the training datas
def create_train_vectors(df_train,vec):
    x_train = np.zeros([len(df_train['Sentences']), len(vec)])
    y_train = np.zeros(len(df_train['Label']))
    # print(df_train['Label'])
    for i in range(len(df_train['Label'])):
        x_train[i,:] = encode_sentence(df_train['Sentences'][i],vec)
        y_train[i] = df_train['Label'][i]
    return x_train,y_train


def train_on_datas(df_train):
    occurences,percentages = compute_words_occurence(df_train)
    vec, sorted_scores = get_best_scores(scores_from_occurences(occurences,percentages))
    x_train,y_train = create_train_vectors(df_train,vec)
    classifier = linear_model.LogisticRegression(max_iter=150)
    classifier.fit(x_train, y_train)
    return classifier,vec
# classifier,vec = train_on_datas(df_train)
#Training the SVM
# classifier = svm.SVC(kernel='sigmoid',probability=True)
# classifier = ensemble.RandomForestClassifier()


# Cross validation


