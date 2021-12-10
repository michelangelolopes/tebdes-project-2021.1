import os
import re

from . import preprocessing

'''load'''

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

def save_comments_to_txt_file(comments, episode):
    comments_file = open("samples/comments_ep%d.txt" % episode, 'w')
    count = 1
    for comment in comments:
        if comment != '':
            #comments_file.write("[%d] %s\n" % (count, comment))
            comments_file.write("%s\n" % comment)
            count += 1
    comments_file.close()

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

    preprocessed_comments = preprocessing.preprocessing_comments(comments)

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
