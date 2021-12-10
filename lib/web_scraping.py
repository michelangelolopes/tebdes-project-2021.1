from bs4 import BeautifulSoup as BS
from optparse import OptionParser
import os
import pickle
import re
import requests
#import bs4
#import pandas as pd

'''options'''

def parse_options():
    parser = OptionParser()
    parser.add_option("-f", "--get-forum-pages", action="store_true", dest="get_forum_pages", default=False, help="get all html pages from forum URL")
    parser.add_option("-s", "--save-html", action="store_true", dest="save_html", default=False, help="save the html pages in html files")
    parser.add_option("-H", "--load-html", action="store_true", dest="load_html", default=False, help="load the html files")
    parser.add_option("-p", "--save-pickle", action="store_true", dest="save_pickle", default=False, help="save the html pages in a list and save the list in a pickle file")
    parser.add_option("-P", "--load-pickle", action="store_true", dest="load_pickle", default=False, help="load the pickle file")
    parser.add_option("-c", "--get-comments-forum", action="store_true", dest="get_comments_forum", default=False, help="get all comments in the html pages")
    parser.add_option("-t", "--save-text", action="store_true", dest="save_text", default=False, help="save the comments in a text file")
    parser.add_option("-T", "--load-text", action="store_true", dest="load_text", default=False, help="load the comments from a text file")
    parser.add_option("-C", "--classify-comments", action="store_true", dest="classify_comments", default=False, help="open/create a csv file that receives comments and opinions inputs")
    parser.add_option("-v", "--verbose", action="store_true", dest="verbose", default=False, help="print debug messages")

    return parser.parse_args()

def run_options(episode, topicid):
    comments = []
    forum_pages = []
    (opt, _) = parse_options()

    if opt.get_forum_pages == True:
        forum_pages = get_forum_pages_from_web(topicid)
    
    if opt.load_pickle == True:
        forum_pages = load_forum_pages_from_pickle_file(episode)
    
    if opt.save_pickle == True:
        save_forum_pages_to_pickle_file(forum_pages, episode)

    if opt.get_comments_forum == True:
        comments = get_comments_from_forum_pages(forum_pages)

    if opt.load_text == True:
        comments = load_comments_from_txt_file(episode)

    if opt.save_text == True:
        #preprocessing_comments(comments)
        save_comments_to_txt_file(comments, episode)

    if opt.classify_comments == True:
        classify_comments_to_csv_file(comments, episode)

    return comments

'''get'''

def get_forum_pages_from_web(topicid):
    session = requests.Session()
    forum_pages = []

    pages = 1
    show = 0
    count = 0

    while count < pages:
        show = count * 50 # são 50 comentários por página
        episode_discussion = session.get(url = "https://myanimelist.net/forum/?topicid=%d&show=%d" % (topicid, show))
        
        if count == 0:
            pages = get_forum_pages_count_from_html_page(episode_discussion.content)
        
        forum_pages.append(episode_discussion.text)
        count += 1
    
    return forum_pages

def get_forum_pages_count_from_html_page(first_page):
    parsed_html = BS(first_page, 'html.parser') #faz o parse da primeira página em html, para poder utilizar a função find da lib BeautifulSoup
    
    html_element = parsed_html.find('div', attrs = {'class': 'pb4'}).text #a div com class = pd4 possui a quantidade total de páginas

    pages_pattern = re.compile('Pages\s\([0-9]*\)') #o padrão no html para encontrar a quantidade de páginas do fórum

    pages = pages_pattern.search(html_element).group(0) #pega a primeira ocorrência do padrão

    pages = int(re.findall('\d+', pages)[0]) #pega o valor numérico da página e converte em inteiro

    return pages

def get_comments_from_forum_pages(forum_pages):
    comments = []

    for page_html in forum_pages:
        comments += get_comments_from_html_page(page_html)

    comments.pop(0) #primeiro comentário é mensagem automática do fórum
    return comments

def get_comments_from_html_page(page_html):
    parsed_html = BS(page_html, 'html.parser')

    body_html = False

    comments = []

    for parent_tag in parsed_html.find_all():
        #a partir da tag body, todas as tags table que seguem, pode conter comentários
        if parent_tag.name == 'body': 
            body_html = True
        
        if body_html == True and parent_tag.name == 'table':
            children_tags = [child_tag for child_tag in parent_tag.find_all() if child_tag.name == 'div']

            for child_tag in children_tags:
                div_content = get_comment_from_html_tag(child_tag)
                
                if div_content != None:
                    comments.append(div_content)
    return comments

def get_comment_from_html_tag(tag):
    find_response = tag.find('div', attrs = {'class': ['clearfix', 'word-break"']})
    if find_response != None:
        div_content = find_response.text #pega todos os textos da div
        
        quotes = find_response.find('div', attrs = {'class': ['quotetext']}) #descobre se a div tem citações
        
        if quotes != None:
            div_content = div_content.replace(quotes.text, '') #retira a citação do comentário
        
        div_content = div_content.split('\n') #divide as linhas do comentário
        div_content = " ".join(div_content) #junta todas as linhas em uma string só novamente, divididas por espaço
    else:
        div_content = None
        
    return div_content

'''load'''

def load_forum_pages_from_pickle_file(episode):
    pickle_file = open("samples/forum_pages_ep%d.pickle" % episode, 'rb')
    forum_pages = pickle.load(pickle_file)
    pickle_file.close()
    return forum_pages

def load_comments_from_txt_file(episode):
    comments_file = open("samples/comments_ep%d.txt" % episode, "r")
    comments = comments_file.readlines()
    comments_file.close()
    return comments

def load_classified_comments_from_csv_file(filename):
    comments_file = open(filename, "r")
    comments_classified = comments_file.readlines() #ignore first line, headers
    comments_file.close()

    comments_classified.pop(0)

    for comment_index in range(0, len(comments_classified)): 
        comment = comments_classified[comment_index].replace("\n", "").split(";")
        comments_classified[comment_index] = (comment[0], comment[1])

    return comments_classified

'''save'''

def save_forum_pages_to_pickle_file(forum_pages, episode):
    pickle_file = open("samples/forum_pages_ep%d.pickle" % episode, 'wb')
    pickle.dump(forum_pages, pickle_file)
    pickle_file.close()

def save_comments_to_txt_file(comments, episode):
    comments_file = open("samples/comments_ep%d.txt" % episode, 'w')
    count = 1
    for comment in comments:
        if comment != '':
            #comments_file.write("[%d] %s\n" % (count, comment))
            comments_file.write("%s\n" % comment)
            count += 1
    comments_file.close()

'''print'''

def print_comments(comments):
    count = 1
    for comment in comments:
        print("[%d]" % count, comment)
        count += 1

def print_opinions(opinions):
    print("-----------------------------------------------------------------------------------------")
    for opinion_index in range(0, len(opinions)):
        print("(%d): %s" % (opinion_index, opinions[opinion_index]), end=" \\/ ")
    print()
    print("-----------------------------------------------------------------------------------------")

'''classify'''

def classify_comments_to_csv_file(comments, episode):
    csv_filepath = "samples/comments_ep%d_classified.csv" % episode

    if os.path.isfile(csv_filepath) == False:
        csv_file = open(csv_filepath, "w").close()

    csv_file = open(csv_filepath, "r+")
    line_count = sum(1 for _ in csv_file) - 1

    if line_count == -1:
        csv_file.write("Comment;Opinion\n")
        line_count = 0

    print(line_count)
    input()
    opinions = ["perfect", "hyped", "good", "average", "bad", "disappointed", "terrible", "irrelevant"]

    preprocessed_comments = preprocessing_comments(comments)

    for i in range(line_count, len(comments)):
        comment = comments[i]
        print("[%d]" % i, comment, end="")
        comment = re.sub('\[[0-9]+\] ', '', comment)
        comment = comment.replace("\n", "")

        print_opinions(opinions)
        
        user_input = input()
        if user_input >= "0" and user_input <= "7":
            user_opinion = opinions[int(user_input)]
        else:
            print("ERROR")

        csv_file.write(preprocessed_comments[i] + ";" + user_opinion + "\n")
        csv_file.flush()
        os.fsync(csv_file)
    csv_file.close()

'''others'''
def preprocessing_comments(comments):
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
