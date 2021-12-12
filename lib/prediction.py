from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, ConfusionMatrixDisplay
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
import matplotlib.pyplot as plt
import os
import pandas as pd
import warnings

from . import dataset
from . import preprocessing

#removendo warning de usar a função coef_ do sklearn para o multinomial naive bayes
warnings.simplefilter(action='ignore', category=FutureWarning) 

'''run'''

def run_classifiers(filename, file_version="", episode=12, remove_features=[], kfolds=4):
    comments_classified = get_preprocessed_data(episode, file_version, remove_features)
    comments = get_cross_validation_kfolds_data(comments_classified, kfolds)

    for fold in range(0, kfolds):
        train_data = comments['train']['data'][fold]
        train_features = comments['train']['features'][fold]
        test_data = comments['test']['data'][fold]
        test_features = comments['test']['features'][fold]

        df_tf_idf_vector_train, df_tf_idf_vector_test, tokens = get_train_test_tf_idf_vector(train_data, test_data)

        filename_fold = filename + "fold_%d/" % fold

        if os.path.exists(filename_fold) == False:
            os.makedirs(filename_fold)

        save_features_count_to_txt_file(train_features, test_features, filename_fold)

        classifiers = get_classifiers_functions()
        classifier_has_coefficient = ["naive_bayes", "svm"]

        classifiers_pred_features = []
        classifiers_tokens_coefficient = []

        for classifier_name, classifier_method in classifiers:
            pred_features, tokens_coefficient = train_classifier(classifier_method, df_tf_idf_vector_train, train_features, df_tf_idf_vector_test)
            save_classifier_confusion_matrix_to_png_file(classifier_name, pred_features, test_features, filename_fold)

            if classifier_name in classifier_has_coefficient:
                classifiers_tokens_coefficient.append((classifier_name, tokens_coefficient))
            classifiers_pred_features.append((classifier_name, pred_features))
        
        save_classifiers_tokens_coefficient_to_txt_file(classifiers_tokens_coefficient, tokens, filename_fold)
        save_classifiers_predictions_to_txt_file(classifiers_pred_features, test_features, filename_fold)
        save_classifiers_metrics_to_txt_file(classifiers_pred_features, test_features, filename_fold)

'''train'''

def train_classifier(classifier, df_tf_idf_vector_train, train_features, df_tf_idf_vector_test):
    trained_classifier = classifier.fit(df_tf_idf_vector_train, train_features)
    pred_features = trained_classifier.predict(df_tf_idf_vector_test)

    tokens_coefficient = ''

    if hasattr(classifier, 'coef_'): #somente naive bayes e svm possuem
        tokens_coefficient = classifier.coef_[0]

    return pred_features, tokens_coefficient

'''save'''

def save_features_count_to_txt_file(train_features, test_features, filename):
    filename = get_filename_with_next_count(filename + "classifiers_features.txt")

    original_features = train_features + test_features
    original_each_feature_count = get_each_feature_count(original_features)
    train_each_feature_count = get_each_feature_count(train_features)
    test_each_feature_count = get_each_feature_count(test_features)

    with open(filename, "w") as file:
        original_str = "ORIGINAL dataset: %d comments\n%s\n\n" % (len(original_features), str(original_each_feature_count))
        train_str = "TRAIN dataset: %d comments\n%s\n\n" % (len(train_features), str(train_each_feature_count))
        test_str = "TEST dataset: %d comments\n%s\n\n" % (len(test_features), str(test_each_feature_count))

        file.write(original_str)
        file.write(train_str)
        file.write(test_str)

def save_classifier_confusion_matrix_to_png_file(classifier_name, pred_features, test_features, filename):
    filename = get_filename_with_next_count(filename + "confusion_matrix_%s.png" % classifier_name)
    
    ConfusionMatrixDisplay.from_predictions(test_features, pred_features)
    plt.gcf().set_size_inches((10, 10), forward=False)
    plt.savefig(filename)
    plt.close()

def save_classifiers_metrics_to_txt_file(classifiers_pred_features, test_features, filename):
    filename = get_filename_with_next_count(filename + "classifiers_metrics.txt")
    
    metric_list = lambda metric: [round(value, 2) for value in metric]

    with open(filename, "w") as file:
        for classifier_name, pred_features in classifiers_pred_features:
            accuracy = accuracy_score(test_features, pred_features)
            precision = precision_score(test_features, pred_features, average=None, zero_division=1)
            recall = recall_score(test_features, pred_features, average=None, zero_division=1)
            f1 = f1_score(test_features, pred_features, average=None, zero_division=1)

            file.write(classifier_name + "\n")
            file.write('Acurácia: %f\n' % accuracy)
            file.write('Precisão: %s\n' % metric_list(precision))
            file.write('Recall: %s\n' % metric_list(recall))
            file.write('Média F1: %s\n\n' % metric_list(f1))

def save_classifiers_predictions_to_txt_file(classifiers_pred_features, test_features, filename):
    filename = get_filename_with_next_count(filename + "classifiers_predictions.txt")
    
    with open(filename, "w") as file:
        count = 0
        for classifier_name, pred_features in classifiers_pred_features:
            file.write(classifier_name + "\n")
            for feature_index in range(0, len(pred_features)):
                string = "PRED, TEST -> %s, %s\n" % (pred_features[feature_index], test_features[feature_index])
                count += 1
                file.write(string)
            file.write("\n")

def save_classifiers_tokens_coefficient_to_txt_file(classifiers_tokens_coefficient, tokens, filename, quantity=20):
    filename = get_filename_with_next_count(filename + "classifiers_words_coefficients.txt")

    with open(filename, "w") as file:
        for classifier_name, tokens_coefficient in classifiers_tokens_coefficient:
            tokens_coefficient = list(tokens_coefficient)

            for token_index in range(0, len(tokens_coefficient)):
                tokens_coefficient[token_index] = (round(float(tokens_coefficient[token_index]), 3), token_index)

            tokens_coefficient.sort(reverse=True)
            tokens_coefficient = tokens_coefficient[0:quantity]

            file.write(classifier_name + "\n")
            for token_value, token_index in tokens_coefficient:
                coefficient_name_value = "%s %.3f" % (tokens[token_index], token_value)
                file.write(coefficient_name_value + "\n")
            file.write("\n")

'''get'''

def get_train_test_tf_idf_vector(train_data, test_data):
    tf_idf_vectorizer = TfidfVectorizer(use_idf=True)

    tf_idf_train = tf_idf_vectorizer.fit_transform(train_data)
    tf_idf_test = tf_idf_vectorizer.transform(test_data)
    tf_idf_tokens = tf_idf_vectorizer.get_feature_names_out()
    
    df_tf_idf_vector_train = pd.DataFrame(data = tf_idf_train.toarray(),columns = tf_idf_tokens)
    df_tf_idf_vector_test = pd.DataFrame(data = tf_idf_test.toarray(),columns = tf_idf_tokens)

    return df_tf_idf_vector_train, df_tf_idf_vector_test, tf_idf_tokens

def get_preprocessed_data(episode, file_version, remove_features):
    filename = "samples/comments_ep%d_classified%s.csv" % (episode, file_version)

    comments_classified = dataset.load_classified_comments_from_csv_file(filename)
    comments_classified = preprocessing.remove_features_from_comments_classified(comments_classified, remove_features)
    comments_classified = preprocessing.remove_stopwords_from_comments_classified(comments_classified)
    comments_classified = preprocessing.use_radicals_from_comments_classified(comments_classified)

    return comments_classified

def get_cross_validation_kfolds_data(comments_classified, kfolds):
    four_folds = KFold(n_splits=kfolds, shuffle=False)
    comments = {}

    comments['train'] = {}
    comments['train']['data'] = []
    comments['train']['features'] = []

    comments['test'] = {}
    comments['test']['data'] = []
    comments['test']['features'] = []

    count = 0
    for train_indexes, test_indexes in four_folds.split(comments_classified):
        comments['train']['data'].append([])
        comments['train']['features'].append([])
        comments['test']['data'].append([])
        comments['test']['features'].append([])

        train_data = [comments_classified[index] for index in train_indexes]
        test_data = [comments_classified[index] for index in test_indexes]

        for comment, feature in train_data:
            comments['train']['data'][count].append(comment)
            comments['train']['features'][count].append(feature)
        
        for comment, feature in test_data:
            comments['test']['data'][count].append(comment)
            comments['test']['features'][count].append(feature)
        count += 1

    return comments

def get_classifiers_functions():
    classifiers = []
    #classifiers.append(("naive_bayes", GaussianNB()))
    classifiers.append(("naive_bayes", MultinomialNB(alpha=0.05)))
    classifiers.append(("svm", svm.SVC(kernel='linear', C=1.0)))
    classifiers.append(("kmeans", KNeighborsClassifier(n_neighbors=4)))
    classifiers.append(("knn", NearestCentroid()))

    return classifiers

def get_each_feature_count(comments_classified):
    feature_count = {}
    for feature in comments_classified:
        if feature not in feature_count:
            feature_count[feature] = 1
        else:
            feature_count[feature] += 1
    
    feature_count = sorted((key, value) for (key,value) in feature_count.items())
    return feature_count

def get_filename_with_next_count(initial_filename):
    count = 0
    filename = initial_filename.replace("###", str(count))
    
    while os.path.exists(filename):
        count += 1
        filename = initial_filename.replace("###", str(count))
    
    return filename
