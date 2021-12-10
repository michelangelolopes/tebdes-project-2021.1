from enum import unique
from optparse import OptionParser
import os
import re
import math
import nltk
import random
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.metrics import recall_score, f1_score, precision_score
from sklearn import svm
import pandas as pd
import matplotlib.pyplot as plt
import time
from lib import web_scraping
from lib import dataset_optimization
from lib import unused

def split_dataset(comments_classified):
    total = len(comments_classified)

    test = int(total/4)
    train = total - test

    comments_classified_train = []
    comments_classified_test = []

    for comment_index in range(0, train):
        comments_classified_train.append(comments_classified[comment_index])
    
    for comment_index in range(train, train + test):
        comments_classified_test.append(comments_classified[comment_index])

    return comments_classified_train, comments_classified_test

# main function
def main():
    episode = 12
    topicid = 1901518 #id da página do fórum para discussão do episódio 12
    #topicid = 1880084 #episódio 1

    #web_scraping.get_forumPages(topicid)
    comments = web_scraping.run_options(episode, topicid)
    web_scraping.print_comments(comments)
    
    return

def project():
    episode = 12
    #random_order_comments(episode)
    #bigger_first_comments(episode)
    #comments_classified = load_csvFile_comments("samples/comments_ep%d_classified.csv" % episode)
    comments_classified = web_scraping.load_classified_comments_from_csv_file("samples/comments_ep%d_classified_best.csv" % episode)
    comments_classified = optimization.remove_irrelevant(comments_classified)
    comments_classified_train, comments_classified_test = split_dataset(comments_classified)

    textos_treino = []
    classes_treino = []
    textos_teste = []
    classes_teste = []

    for comment, group in comments_classified_train:
        textos_treino.append(comment)
        classes_treino.append(group)

    for comment, group in comments_classified_test:
        textos_teste.append(comment)
        classes_teste.append(group)

    tfidfvectorizer = TfidfVectorizer(use_idf=True)

    tfidf_treino = tfidfvectorizer.fit_transform(textos_treino)
    tfidf_teste = tfidfvectorizer.transform(textos_teste)
    tfidf_tokens = tfidfvectorizer.get_feature_names_out()
    
    # print(len(tfidf_tokens))
    #print(len(tfidf_treino.toarray()[2]))
    df_tfidfvect_treino = pd.DataFrame(data = tfidf_treino.toarray(),columns = tfidf_tokens)
    df_tfidfvect_teste = pd.DataFrame(data = tfidf_teste.toarray(),columns = tfidf_tokens)

    # print("\nTD-IDF Vectorizer\n")
    # print(df_tfidfvect_treino)
    # print(df_tfidfvect_teste)

    def train(nome_metodo, classificador, colecao_treino, classes_treino, colecao_teste,  classes_teste):
        #print(nome_metodo)
        classificador.fit(colecao_treino, classes_treino)
        classes_pred = classificador.fit(colecao_treino, classes_treino).predict(colecao_teste)

        metric_list = lambda metric: [round(value, 2) for value in metric]
        #matriz_confusao = confusion_matrix(classes_teste, classes_pred)
        acuracia = classificador.score(colecao_teste, classes_teste)
        precisao = precision_score(classes_teste, classes_pred, average=None, zero_division=1)
        recall = recall_score(classes_teste, classes_pred, average=None, zero_division=1)
        f1 = f1_score(classes_teste, classes_pred, average=None, zero_division=1)

        #print('Acurácia: %s' % acuracia)
        #print('Precisão: %s' % metric_list(precisao))
        #print('Recall: %s' % metric_list(recall))
        #print('Média F1: %s' % metric_list(f1))
        #print()
        #ConfusionMatrixDisplay.from_predictions(classes_pred, classes_teste)
        #plt.show()
        #print(matriz_confusao)
        return acuracia

    sum = 0

    sum += train("Naive Bayes", MultinomialNB(alpha=0.05), df_tfidfvect_treino, classes_treino, df_tfidfvect_teste, classes_teste)

    sum += train("SVM", svm.SVC(kernel='linear', C=1.0), df_tfidfvect_treino, classes_treino, df_tfidfvect_teste, classes_teste)

    sum += train("K-means", KNeighborsClassifier(n_neighbors=3), df_tfidfvect_treino, classes_treino, df_tfidfvect_teste, classes_teste)

    sum += train("Knn", NearestCentroid(), df_tfidfvect_treino, classes_treino, df_tfidfvect_teste, classes_teste)

    return sum
    '''
    term_document_train = term_document_matrix(comments_classified_train)
    print(term_document_train)


    naiveBayes = nltk.NaiveBayesClassifier.train(term_document_train)
    radicals_test, unique_test = get_unique_words_radicals(comments_classified_test)

    comments_test = [comment for comment, _ in radicals_test]
    correct_classify = 0

    for comment_index in range(0, len(comments_test)):
        sentence_test = characteristic_vector(comments_test[comment_index], unique_test)
        #print(comments_classified_test[comment_index])
        if naiveBayes.classify(sentence_test) == comments_classified_test[comment_index][1]:
            correct_classify += 1

    print("Correct:", (correct_classify/len(comments_test)))
    print(naiveBayes.show_most_informative_features())
    print(len(comments_classified_train), len(comments_classified_test))
    '''
    '''
    comments = []
    groups = []

    for comment, group in comments_classified:
        comments.append(comment)
        groups.append(group)

    # unique_words_comments = get_unique_words(comments_classified)
    # tf_idf = get_tf_idf_matrix(comments, unique_words_comments)

    tfidfvectorizer = TfidfVectorizer(use_idf=True)

    tfidf_treino = tfidfvectorizer.fit_transform(comments)
    print(tfidf_treino)
    tfidf_tokens = tfidfvectorizer.get_feature_names_out()
    print(tfidf_tokens)
    tf_idf = pd.DataFrame(data = tfidf_treino.toarray(), columns = tfidf_tokens)
    knn = NearestCentroid().fit(tf_idf, groups)

    print(len(knn.centroids_))
    '''

def optimization():
        initial_time = time.time()
        run_time = 0
        max_value = 0
        minute = 1

        while run_time < (60 * 5):
            value = project()
            
            if max_value < value:
                print(value)
                max_value = value
                input()
                old_file = os.getcwd() + "/samples/comments_ep12_classified_random.csv"
                new_file = os.getcwd() + "/samples/comments_ep12_classified_best.csv"
                
                if os.path.exists("./samples/comments_ep12_classified_best.csv"):
                    os.remove("./samples/comments_ep12_classified_best.csv")
                os.rename(old_file, new_file)

            cur_time = time.time()
            run_time = cur_time - initial_time

            if run_time > (minute * 60):
                print(round(run_time, 2), "seconds")
                minute += 1

if __name__ == "__main__":
    main()
    #optimization()