from lib import optimization
from lib import prediction
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# main function
def main():
    #episode = 12
    #topicid = 1901518 #id da página do fórum para discussão do episódio 12
    #topicid = 1880084 #episódio 1

    remove_features = ["irrelevant", "good", "terrible", "perfect"]
    original_filename = prediction.get_correct_filename("results/execution ###/")
    original_filename += "DATASET/FEATURES/"

    #dataset original

    ###todas as classes
    filename = original_filename.replace("DATASET", "original_dataset").replace("FEATURES", "all_features")
    prediction.run_classifiers(filename=filename)
    
    ###metade das classes
    filename = original_filename.replace("DATASET", "original_dataset").replace("FEATURES", "half_features")
    prediction.run_classifiers(remove_features=remove_features, filename=filename)

    #dataset otimizado

    ###todas as classes
    filename = original_filename.replace("DATASET", "otimized_dataset").replace("FEATURES", "all_features")
    prediction.run_classifiers(file_version="_best", filename=filename)

    ###metade das classes
    filename = original_filename.replace("DATASET", "otimized_dataset").replace("FEATURES", "half_features")
    prediction.run_classifiers(file_version="_best", remove_features=remove_features, filename=filename)
    
    return

if __name__ == "__main__":
    main()
    #optimization.optimize_train_test_datasets(10)    
