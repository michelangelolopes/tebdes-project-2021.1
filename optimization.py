from sys import argv
import os
import random
import time

from lib import prediction

'''optimize'''

def optimize_train_test_datasets(minutes, episode=12):
    initial_time = time.time()
    run_time = 0
    max_value = 0
    minute_count = 1

    if os.path.exists("./samples/comments_ep12_classified_best.csv"):
        max_value = get_classifiers_sum_acuracies(file_version="_best")
        max_value += get_classifiers_sum_acuracies(file_version="_best", remove_features=["irrelevant", "good", "terrible", "perfect"])
        print(max_value)

    while run_time < (60 * minutes):
        save_comments_random_ordered_to_csv_file(episode)
        value = get_classifiers_sum_acuracies(file_version="_random")
        value += get_classifiers_sum_acuracies(file_version="_random", remove_features=["irrelevant", "good", "terrible", "perfect"])
        
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

        if run_time > (minute_count * 60):
            print(round(run_time, 2), "seconds")
            minute_count += 1

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

def get_classifiers_sum_acuracies(file_version, episode = 12, remove_features=[], kfolds=4):
    comments_classified = prediction.get_preprocessed_data(episode, file_version, remove_features)
    comments = prediction.get_cross_validation_kfolds_data(comments_classified, kfolds)

    sum = 0
    for fold in range(0, kfolds):
        train_data = comments['train']['data'][fold]
        train_features = comments['train']['features'][fold]
        test_data = comments['test']['data'][fold]
        test_features = comments['test']['features'][fold]

        df_tf_idf_vector_train, df_tf_idf_vector_test, _ = prediction.get_train_test_tf_idf_vector(train_data, test_data)

        classifiers = prediction.get_classifiers_functions()

        for _, classifier_method in classifiers:
            pred_features, _ = prediction.train_classifier(classifier_method, df_tf_idf_vector_train, train_features, df_tf_idf_vector_test)
            sum += prediction.accuracy_score(test_features, pred_features)
    
    return sum

if __name__ == "__main__":
    running_time = 3

    try:
        running_time = int(argv[1])
    except ValueError:
        print("\"%s\" não é um valor inteiro maior que zero, então será utilizado o tempo de execução padrão de 3 min." % argv[1])
    
    if running_time <= 0:
        print("%d não é um valor inteiro maior que zero, então será utilizado o tempo de execução padrão de 3 min." % running_time)
        running_time = 3
    else:
        print("A otimização do conjunto de dados será executada por %d min." % running_time)

    optimize_train_test_datasets(running_time)