from lib import prediction
from lib import results

# main function
def main():
    #episode, topicid = 12, 1901518
    #episode, topicid = 1, 1880084

    remove_features = ["irrelevant", "good", "terrible", "perfect"]
    # remove_features = ["disappointed", "hyped", "terrible", "perfect", "average", "bad"]
    original_filename = prediction.get_filename_with_next_count("outputs/execution_###/")
    original_filename += "DATASET/FEATURES/"

    #dataset original

    print("running case: original_dataset + all_features")
    ###todas as classes
    filename = original_filename.replace("DATASET", "original_dataset").replace("FEATURES", "all_features")
    prediction.run_classifiers(filename=filename)
    
    print("running case: original_dataset + half_features")
    ###metade das classes
    filename = original_filename.replace("DATASET", "original_dataset").replace("FEATURES", "half_features")
    prediction.run_classifiers(remove_features=remove_features, filename=filename)

    #dataset otimizado

    print("running case: otimized_dataset + all_features")
    ###todas as classes
    filename = original_filename.replace("DATASET", "otimized_dataset").replace("FEATURES", "all_features")
    prediction.run_classifiers(file_version="_best", filename=filename)

    print("running case: otimized_dataset + half_features")
    ###metade das classes
    filename = original_filename.replace("DATASET", "otimized_dataset").replace("FEATURES", "half_features")
    prediction.run_classifiers(file_version="_best", remove_features=remove_features, filename=filename)

if __name__ == "__main__":
    main()
    results.save_all_executions_accuracies_to_csv_files()
