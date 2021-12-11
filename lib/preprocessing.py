import nltk
import re

'''remove'''

def remove_noise_data_from_comments(comments):
    preprocessed_comments = list(comments)

    for index in range(0, len(preprocessed_comments)):
        comment = preprocessed_comments[index]
        comment = comment.lower() #coloca todas as palavras em letra minúscula
        comment = re.sub(r'@\w+', '', comment) #remove citações de usuário
        comment = re.sub(r'#\w+', '', comment) #remove hashtags
        comment = re.sub('https?://[a-z0-9./\-_]+', '', comment) #remove links
        comment = re.sub('\[[a-z0-9./\- ]+\]', '', comment) #remove texto dentro de colchetes, por normalmente conter info de marcação
        comment = re.sub(';', ',', comment) #trocando ponto e vírgula por vírgula, para utilizar csv depois
        #div_content = re.sub('[^A-Za-z0-9 !.?:,()\{\}\[\]-_+=\']+', ' ', div_content) #aceita apenas caracteres normais e números
        comment = re.sub('[^a-z0-9 \']+', ' ', comment) #aceita apenas letras e números
        comment = re.sub(' +', ' ', comment) #troca múltiplos espaços por um espaço apenas
        comment = re.sub('^ ', '', comment) #tira espaço do começo da string
        comment = re.sub(' $', '', comment) #tira espaço do final da string

        preprocessed_comments[index] = comment
    return preprocessed_comments

def remove_features_from_comments_classified(comments_classified, remove_features):
    new = []

    for comment_index in range(0, len(comments_classified)):
        if comments_classified[comment_index][1] not in remove_features:
            new.append(comments_classified[comment_index])

    return new

def remove_stopwords_from_comments_classified(sentences): 
    stopwords_list = nltk.corpus.stopwords.words('english')
    sentences_nostopwords = []
    
    for (sentence, group) in sentences:
        sentence_nostopwords = [word for word in sentence.split() if word not in stopwords_list]
        sentences_nostopwords.append((' '.join(sentence_nostopwords), group))
    return sentences_nostopwords

'''use'''

def use_radicals_from_comments_classified(sentences): 
    radicals = nltk.stem.RSLPStemmer()
    sentences_radicals = []
    for (sentence, group) in sentences:
        sentence_radicals = [str(radicals.stem(word)) for word in sentence.split()]
        sentences_radicals.append((' '.join(sentence_radicals), group))
    return sentences_radicals