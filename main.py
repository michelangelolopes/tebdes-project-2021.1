from enum import unique
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.metrics import recall_score, f1_score, precision_score
from sklearn import svm
import matplotlib.pyplot as plt
import os
import pandas as pd
import time
from lib import web_scraping
from lib import dataset_optimization
from lib import unused
import nltk

def get_radicals(sentences): 
    #basicamente a mesma função que o professor tinha implementado, mas desconsiderando "classes" e com variáveis renomeadas
    radicals = nltk.stem.RSLPStemmer()
    sentences_radicals = []
    for (sentence, group) in sentences:
        sentence_radicals = [str(radicals.stem(word)) for word in sentence.split()]
        sentences_radicals.append((' '.join(sentence_radicals), group))
    return sentences_radicals

def remove_stopwords(sentences): 
    #basicamente a mesma função que o professor tinha implementado, mas desconsiderando "classes" e com variáveis renomeadas
    stopwords_list = nltk.corpus.stopwords.words('english')
    sentences_nostopwords = []
    
    for (sentence, group) in sentences:
        sentence_nostopwords = [word for word in sentence.split() if word not in stopwords_list]
        sentences_nostopwords.append((' '.join(sentence_nostopwords), group))
    return sentences_nostopwords

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
    dataset_optimization.random_order_comments(episode)
    #bigger_first_comments(episode)
    #comments_classified = load_csvFile_comments("samples/comments_ep%d_classified.csv" % episode)
    comments_classified = web_scraping.load_classified_comments_from_csv_file("samples/comments_ep%d_classified" % episode + file_version + ".csv")
    comments_classified = dataset_optimization.remove_irrelevant(comments_classified)
    comments_classified = remove_stopwords(comments_classified)
    comments_classified = get_radicals(comments_classified)

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

# main function
def main():
    episode = 12
    topicid = 1901518 #id da página do fórum para discussão do episódio 12
    #topicid = 1880084 #episódio 1

    #web_scraping.get_forumPages(topicid)
    comments = web_scraping.run_options(episode, topicid)
    web_scraping.print_comments(comments)
    
    return

def train_classifier(classifier_name, classifier, df_tf_idf_vector_train, train_features, df_tf_idf_vector_test, test_features):
    #print(classifier_name)
    trained_classifier = classifier.fit(df_tf_idf_vector_train, train_features)
    pred_features = trained_classifier.predict(df_tf_idf_vector_test)

    metric_list = lambda metric: [round(value, 2) for value in metric]
    #matriz_confusao = confusion_matrix(test_features, pred_features)
    acuracy = classifier.score(df_tf_idf_vector_test, test_features)
    precision = precision_score(test_features, pred_features, average=None, zero_division=1)
    recall = recall_score(test_features, pred_features, average=None, zero_division=1)
    f1 = f1_score(test_features, pred_features, average=None, zero_division=1)

    #print('Acurácia: %s' % acuracy)
    #print('Precisão: %s' % metric_list(precision))
    #print('Recall: %s' % metric_list(recall))
    #print('Média F1: %s' % metric_list(f1))
    #print(matriz_confusao)
    #print()
    return pred_features

def save_confusion_matrix_to_png_file(classifier_name, pred_features, test_features):
    ConfusionMatrixDisplay.from_predictions(pred_features, test_features)
    count = 1

    filename = "confusion_matrix_" + classifier_name + "_" + str(count) + ".svg"
    
    while os.path.exists(filename):
        count += 1
        filename = "confusion_matrix_" + classifier_name + "_" + str(count) + ".svg"

    plt.show()
    #plt.savefig(filename, format="svg")

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

def get_classifiers():
    classifiers = []
    classifiers.append(("naive_bayes", MultinomialNB(alpha=0.05)))
    classifiers.append(("svm", svm.SVC(kernel='linear', C=1.0)))
    classifiers.append(("kmeans", KNeighborsClassifier(n_neighbors=3)))
    classifiers.append(("knn", NearestCentroid()))

    return classifiers

def run_classifiers(file_version=""):
    episode = 12
    train_data, train_features, test_data, test_features = get_train_test_data_features(episode, file_version)
    df_tf_idf_vector_train, df_tf_idf_vector_test = get_train_test_tf_idf_vector(train_data, test_data)

    classifiers = get_classifiers()

    for classifier_name, classifier_method in classifiers:
        pred_features = train_classifier(classifier_name, classifier_method, df_tf_idf_vector_train, train_features, df_tf_idf_vector_test, test_features)
        save_confusion_matrix_to_png_file(classifier_name, pred_features, test_features)

def run_classifiers_optimization(file_version):
    episode = 12
    train_data, train_features, test_data, test_features = get_train_test_data_features(episode, file_version)
    df_tf_idf_vector_train, df_tf_idf_vector_test = get_train_test_tf_idf_vector(train_data, test_data)

    classifiers = get_classifiers()

    sum = 0
    for _, classifier_method in classifiers:
        sum += dataset_optimization.get_score(classifier_method, df_tf_idf_vector_train, train_features, df_tf_idf_vector_test, test_features)
    
    return sum

def run_optimization():
    initial_time = time.time()
    run_time = 0
    max_value = 0
    minute = 1

    if os.path.exists("./samples/comments_ep12_classified_best.csv"):
        max_value = run_classifiers_optimization("_best")
        print(max_value)

    while run_time < (60 * 5):
        value = run_classifiers_optimization("_random")
        
        if max_value < value:
            print(value)
            max_value = value
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
    #main()
    run_classifiers("_best")
    #run_optimization()
