from lib import prediction

# main function
def main():
    #episode = 12
    #topicid = 1901518 #id da página do fórum para discussão do episódio 12
    #topicid = 1880084 #episódio 1

    prediction.run_classifiers(file_version="_best")
    
    return

if __name__ == "__main__":
    main()
