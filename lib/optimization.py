import os
import random
import time

from . import prediction

'''optimize'''

def optimize_train_test_datasets():
    initial_time = time.time()
    run_time = 0
    max_value = 0
    minute = 1
    episode = 12

    if os.path.exists("./samples/comments_ep12_classified_best.csv"):
        max_value = get_classifiers_sum_acuracies("_best")
        print(max_value)
        input()

    while run_time < (60 * 5):
        save_comments_random_ordered_to_csv_file(episode)
        value = get_classifiers_sum_acuracies("_random")
        
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

'''save'''

def save_comments_random_ordered_to_csv_file(episode):
    comments_file = open("samples/comments_ep%d_classified.csv" % episode, "r")
    comments_classified = comments_file.readlines() #ignore first line, headers
    comments_file.close()

    headers = comments_classified.pop(0)

    randomized_comments = []
    comments_indexes = []

    for _ in range(0, len(comments_classified)):
        random_index = random.randrange(0, len(comments_classified))

        while random_index in comments_indexes:
            random_index = random.randrange(0, len(comments_classified))

        if random_index not in comments_indexes:
            comments_indexes.append(random_index)
            randomized_comments.append(comments_classified[random_index])

    comments_classified = randomized_comments

    comments_classified.insert(0, headers)

    with open("samples/comments_ep%d_classified_random.csv" % episode, "w") as new_file:
        for comment in comments_classified:
            new_file.write(comment)

'''get'''

def get_classifiers_sum_acuracies(file_version, episode = 12):
    train_data, train_features, test_data, test_features = prediction.get_train_test_data_features(episode, file_version)
    df_tf_idf_vector_train, df_tf_idf_vector_test = prediction.get_train_test_tf_idf_vector(train_data, test_data)

    classifiers = prediction.get_classifiers()

    sum = 0
    for _, classifier_method in classifiers:
        sum += get_classifier_accuracy(classifier_method, df_tf_idf_vector_train, train_features, df_tf_idf_vector_test, test_features)
    
    return sum

def get_classifier_accuracy(classifier, df_tf_idf_vector_data, train_features, df_tf_idf_vector_test, test_features):
    trained_classifier = classifier.fit(df_tf_idf_vector_data, train_features)
    trained_classifier.predict(df_tf_idf_vector_test)
    accuracy = classifier.score(df_tf_idf_vector_test, test_features)
    
    return accuracy
