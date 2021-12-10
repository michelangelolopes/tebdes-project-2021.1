from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import recall_score, f1_score, precision_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
import matplotlib.pyplot as plt
import os
import pandas as pd

from . import dataset
from . import preprocessing

'''run'''

def run_classifiers(file_version="", episode=12):
    train_data, train_features, test_data, test_features = get_train_test_data_features(episode, file_version)
    df_tf_idf_vector_train, df_tf_idf_vector_test = get_train_test_tf_idf_vector(train_data, test_data)

    classifiers = get_classifiers()

    for classifier_name, classifier_method in classifiers:
        pred_features = train_classifier(classifier_name, classifier_method, df_tf_idf_vector_train, train_features, df_tf_idf_vector_test, test_features)
        open_confusion_matrix_image(classifier_name, pred_features, test_features)

'''open'''

def open_confusion_matrix_image(classifier_name, pred_features, test_features):
    ConfusionMatrixDisplay.from_predictions(pred_features, test_features)
    
    filename = "./results/" + classifier_name + ".png"
    
    if os.path.exists(filename) == False:
        plt.show()
        
    #plt.savefig(filename)

'''train'''

def train_classifier(classifier_name, classifier, df_tf_idf_vector_train, train_features, df_tf_idf_vector_test, test_features):
    trained_classifier = classifier.fit(df_tf_idf_vector_train, train_features)
    pred_features = trained_classifier.predict(df_tf_idf_vector_test)

    metric_list = lambda metric: [round(value, 2) for value in metric]
    #matriz_confusao = confusion_matrix(test_features, pred_features)
    accuracy = classifier.score(df_tf_idf_vector_test, test_features)
    precision = precision_score(test_features, pred_features, average=None, zero_division=1)
    recall = recall_score(test_features, pred_features, average=None, zero_division=1)
    f1 = f1_score(test_features, pred_features, average=None, zero_division=1)

    filename = "./results/classifiers_results.txt"

    if os.path.exists(filename) == False:
        with open(filename, "a+") as file:
            file.write(classifier_name + "\n")
            file.write('Acurácia: %s\n' % accuracy)
            file.write('Precisão: %s\n' % metric_list(precision))
            file.write('Recall: %s\n' % metric_list(recall))
            file.write('Média F1: %s\n\n' % metric_list(f1))
    else:
        print(classifier_name)
        print('Acurácia: %s' % accuracy)
        print('Precisão: %s' % metric_list(precision))
        print('Recall: %s' % metric_list(recall))
        print('Média F1: %s' % metric_list(f1))
        print()

    #print(matriz_confusao)
    return pred_features

'''get'''

def get_train_test_tf_idf_vector(train_data, test_data):
    tf_idf_vectorizer = TfidfVectorizer(use_idf=True)

    tf_idf_train = tf_idf_vectorizer.fit_transform(train_data)
    tf_idf_test = tf_idf_vectorizer.transform(test_data)
    tf_idf_tokens = tf_idf_vectorizer.get_feature_names_out()
    
    # print(len(tf_idf_tokens))
    #print(len(tf_idf_train.toarray()[2]))
    df_tf_idf_vector_train = pd.DataFrame(data = tf_idf_train.toarray(),columns = tf_idf_tokens)
    df_tf_idf_vector_test = pd.DataFrame(data = tf_idf_test.toarray(),columns = tf_idf_tokens)

    # print("\nTD-IDF Vectorizer\n")
    # print(df_tf_idf_vector_train)
    # print(df_tf_idf_vector_test)
    return df_tf_idf_vector_train, df_tf_idf_vector_test

def get_train_test_classified_data(comments_classified):
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

def get_train_test_data_features(episode, file_version):
    filename = "samples/comments_ep%d_classified" % episode + file_version + ".csv"
    comments_classified = dataset.load_classified_comments_from_csv_file(filename)
    comments_classified = preprocessing.remove_irrelevant_feature_from_comments_classified(comments_classified)
    comments_classified = preprocessing.remove_stopwords_from_comments_classified(comments_classified)
    comments_classified = preprocessing.use_radicals_from_comments_classified(comments_classified)

    comments_classified_train, comments_classified_test = get_train_test_classified_data(comments_classified)

    train_data, train_features = [], []
    test_data, test_features = [], []

    for comment, group in comments_classified_train:
        train_data.append(comment)
        train_features.append(group)

    for comment, group in comments_classified_test:
        test_data.append(comment)
        test_features.append(group)

    return train_data, train_features, test_data, test_features

def get_classifiers():
    classifiers = []
    classifiers.append(("naive_bayes", MultinomialNB(alpha=0.05)))
    classifiers.append(("svm", svm.SVC(kernel='linear', C=1.0)))
    classifiers.append(("kmeans", KNeighborsClassifier(n_neighbors=3)))
    classifiers.append(("knn", NearestCentroid()))

    return classifiers
