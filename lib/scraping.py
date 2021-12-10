from bs4 import BeautifulSoup as BS
import pickle
import re
import requests

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

'''save'''

def save_forum_pages_to_pickle_file(forum_pages, episode):
    pickle_file = open("samples/forum_pages_ep%d.pickle" % episode, 'wb')
    pickle.dump(forum_pages, pickle_file)
    pickle_file.close()
