import os
import re

'''get'''

def get_correct_count_filename(initial_filename):
    count = 0
    filename = initial_filename.replace("###", str(count))
    
    while os.path.exists(filename):
        count += 1
        filename = initial_filename.replace("###", str(count))
    count -= 1
    return count

def get_accuracies_dicts():
    original_filename = "outputs/execution_###/"
    executions = get_correct_count_filename(original_filename)
    original_filename += "DATASET/FEATURES/"

    classifier_names = ['naive_bayes', 'svm', 'kmeans', 'knn']
    replace_cases = lambda filename, dataset, features: filename.replace("DATASET", dataset).replace("FEATURES", features)

    cases = [
                ("original_dataset", "all_features"), 
                ("original_dataset", "half_features"), 
                ("otimized_dataset", "all_features"), 
                ("otimized_dataset", "half_features")
            ]

    execution_data_list = []

    for execution in range(0, executions + 1):
        accuracy_dict = {}
        cur_filename = original_filename.replace("###", str(execution))
        
        for dataset, features in cases:
            if dataset not in accuracy_dict:
                accuracy_dict[dataset] = {}
            if features not in accuracy_dict[dataset]:
                accuracy_dict[dataset][features] = []

            case_filename = replace_cases(cur_filename, dataset, features)

            fold_filename = case_filename + "fold_###/"
            folds = get_correct_count_filename(fold_filename)

            for fold in range(0, folds + 1):
                #print("abc")
                #input()
                #accuracy_dict[dataset].append({})

                final_filename = fold_filename.replace("###", str(fold))
                final_filename += "classifiers_metrics.txt"

                with open(final_filename, 'r') as file:
                    file_content = "".join(file.readlines())
                
                accuracy = re.findall(r'Acurácia:\s([0-9.]+)', file_content)

                classifier_data_dict = {}
                for index in range(0, len(classifier_names)):
                    classifier_name = classifier_names[index]
                    classifier_data_dict[classifier_name] = float(accuracy[index])
                accuracy_dict[dataset][features].append(classifier_data_dict)

        execution_data_list.append(accuracy_dict)
    return execution_data_list

def get_accuracies_average_dict(accuracy_dict):
    classifier_names = ['naive_bayes', 'svm', 'kmeans', 'knn']
    accuracies_average_dict = {}

    for dataset in accuracy_dict:
        if dataset not in accuracies_average_dict:
            accuracies_average_dict[dataset] = {}
        for features in accuracy_dict[dataset]:
            if features not in accuracies_average_dict[dataset]:
                accuracies_average_dict[dataset][features] = {}

            for classifier_name in classifier_names:
                if classifier_name not in accuracies_average_dict[dataset][features]:
                    accuracies_average_dict[dataset][features][classifier_name] = 0

                folds_count = len(accuracy_dict[dataset][features])
                for fold in range(0, folds_count):
                    accuracies_average_dict[dataset][features][classifier_name] += accuracy_dict[dataset][features][fold][classifier_name]
                
                if folds_count != 0:
                    accuracies_average_dict[dataset][features][classifier_name] /= folds_count
    return accuracies_average_dict

'''save'''

def save_accuracies_to_csv_file(accuracy_dict, accuracies_average_dict, filename):
    classifier_names = ['naive_bayes', 'svm', 'kmeans', 'knn']

    for dataset in accuracy_dict:
        for features in accuracy_dict[dataset]:
            cur_filename = filename + "accuracies_%s_%s.csv" % (dataset, features)

            with open(cur_filename, 'w') as file:
                file.write(";fold_0;fold_1;fold_2;fold_3;average\n")
                
                for classifier_name in classifier_names:
                    file.write(classifier_name + ";")

                    folds_count = len(accuracy_dict[dataset][features])
                    for fold in range(0, folds_count):
                        value = accuracy_dict[dataset][features][fold][classifier_name]
                        value = round(value, 4)
                        file.write(str(value).replace(".", ",") + ";")
                    
                    average_value = accuracies_average_dict[dataset][features][classifier_name]
                    average_value = round(average_value, 4)
                    file.write(str(average_value).replace(".", ",") + "\n")

def save_all_executions_accuracies_to_csv_files():
    executions_data_list = get_accuracies_dicts()
    original_filename = "outputs/execution_###/"

    count = 0
    for accuracy_dict in executions_data_list:
        filename = original_filename.replace("###", str(count))
        accuracies_average_dict = get_accuracies_average_dict(accuracy_dict)
        save_accuracies_to_csv_file(accuracy_dict, accuracies_average_dict, filename)
        count += 1