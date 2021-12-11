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

    #dataset original

    ###todas as classes
    prediction.run_classifiers(remove_features=[])
    ###sem classe irrelevant
    prediction.run_classifiers()
    ###sem classes listadas
    prediction.run_classifiers(remove_features=remove_features)

    #dataset otimizado

    ###todas as classes
    prediction.run_classifiers(file_version="_best", remove_features=[])
    ###sem classe irrelevant
    prediction.run_classifiers(file_version="_best")
    ###sem classes listadas
    prediction.run_classifiers(file_version="_best", remove_features=remove_features)
    
    return

if __name__ == "__main__":
    main()
    #optimization.optimize_train_test_datasets(3)    
