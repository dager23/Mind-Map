import nltk
from nltk.cluster.util import cosine_distance
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from rake_nltk import Rake

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import math

import requests
from bs4 import BeautifulSoup
import re
import heapq

import ldamodule

#GLOBAL VARIABLES

stop_words = stopwords.words('english')

#WEB SCRAPPING

def scrape_data(URL):
    html_page = requests.get(URL).text
    soup = BeautifulSoup(html_page, 'lxml')
    paraContent = soup.find_all('p')
    paragraph = ""
    for para in paraContent:
        paragraph += para.text
    paragraph = re.sub(r'\[[0-9a-zA-Z]*\]', ' ', paragraph)
    paragraph = re.sub(r"(  |\r|\n|\t)", ' ', paragraph)
    
    return paragraph

def clean_text(text):
    space_pattern = r"(  |\r|\n|\t)"
    citation_pattern = r'\[[0-9a-zA-Z]*\]'
    text = re.sub(space_pattern, " ", text)
    text = re.sub(citation_pattern, "", text)
    
    return text

#TEXT SUMMARIZATION

def get_important_sentences(data):
    sentence_tokens = nltk.sent_tokenize(data)
    stop_words = nltk.corpus.stopwords.words('english')
    word_frequencies = {}
    word_tokens = nltk.word_tokenize(data)
    for word in word_tokens:
        if word not in stop_words:
            if word not in word_frequencies.keys():
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1
                
    # Weighted Frequencies
    maximum_frquency_word = max(word_frequencies.values())
    for word in word_frequencies.keys():
        word_frequencies[word] = (word_frequencies[word]/maximum_frquency_word)
    
    # Sentence Score
    sentence_scores = {}
    for sentence in sentence_tokens:
        for word in nltk.word_tokenize(sentence.lower()):
            if word in word_frequencies.keys():
                if(len(sentence.split(" ")) < 30):
                    if sentence not in sentence_scores.keys():
                        sentence_scores[sentence] = word_frequencies[word]
                    else:
                        sentence_scores[sentence] += word_frequencies[word]
    
    top_sentences = heapq.nlargest(25, sentence_scores, key=sentence_scores.get)
    result = []
    for sentence in top_sentences:
        result.append(nltk.word_tokenize(sentence))
    
    return result

def sentence_similarity(sent1, sent2, stop_words):
    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]
    all_words = list((set(sent1+sent2)))
    
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
    for w in sent1:
        if w not in stop_words:
            vector1[all_words.index(w)] += 1
    for w in sent2:
        if w not in stop_words:
            vector2[all_words.index(w)] += 1
    
    return 1-cosine_distance(vector1, vector2)

def gen_sim_matrix(sentences, stop_words):
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if(idx1 == idx2):
                continue
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)
    
    return similarity_matrix

def generate_summary(data, top_n=5):
    sentences = get_important_sentences(data)
    summarized_text = []
    sentence_similarity_matrix = gen_sim_matrix(sentences, stop_words)
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_matrix)
    scores = nx.pagerank(sentence_similarity_graph)
    ranked_sentences = sorted(((scores[i], s) for i,s in enumerate(sentences)), reverse=True)
    upper_limit = min(len(ranked_sentences), top_n)
    for i in range(upper_limit):
        summarized_text.append(" ".join(ranked_sentences[i][1]))
    summary = " ".join(summarized_text)
    return summary


# GET KEYWORDS

def set_threshold(phrases):
    scores = []
    for phrase in phrases:
        scores.append(phrase[0])
    threshold = np.percentile(scores, 65)
    
    return threshold


def get_common_phrases(phrase_list, max_nodes=5):
    common_phrases = []
    for group1 in phrase_list:
        flag = 0
        count_common_phrases = 0
        for phrase1 in group1:
            for group2 in phrase_list:
                if(group1 != group2):
                    for phrase2 in group2:
                        if(phrase1[1] == phrase2[1]):
                            common_phrases.append([phrase1[1], phrase_list.index(group1), phrase_list.index(group2)])
                            count_common_phrases += 1
                            if(count_common_phrases == max_nodes):
                                flag = 1
                                break
                    if(flag):
                        break
            if(flag):
                break
                       
    return common_phrases

def get_keywords(grouped_text, max_nodes=5):
    phrases_list = []
    for idx in grouped_text:
        if(grouped_text[idx]):
            rake_model = Rake()
            rake_model.extract_keywords_from_text(grouped_text[idx])
            phrases = rake_model.get_ranked_phrases_with_scores()
            phrases_list.append(phrases)
        else:
            phrases_list.append([])
    final_keywords = get_best_phrases(phrases_list, max_nodes)
    common_list = get_common_phrases(phrases_list, max_nodes//2)
    for phrase in common_list:
        keyword = phrase[0]
        if(keyword not in final_keywords[phrase[1]]):
            final_keywords[phrase[1]].append(keyword)
        if(keyword not in final_keywords[phrase[2]]):
            final_keywords[phrase[2]].append(keyword)
    return final_keywords

def get_best_phrases(phrase_list, max_nodes=5):
    final_phrases = []
    for phrases in phrase_list:
        final_phrases_topic = []
        if(phrases):
            threshold = set_threshold(phrases)
            for phrase in phrases:
                if(phrase[0] > math.floor(threshold)):
                    if(final_phrases_topic):
                        flag = 0
                        for prev_phrase in final_phrases:
                            if(prev_phrase):
                                similarity = sentence_similarity(prev_phrase, phrase[1], stop_words)
                                if(similarity > 0.99):
                                    flag = 1
                                    break
                        if(not flag):
                            final_phrases_topic.append(phrase[1])
                    else:
                        final_phrases_topic.append(phrase[1])
        final_phrases.append(final_phrases_topic[:max_nodes])
    return final_phrases

def create_keywords_from_text(text, max_nodes=5, sentence_group=3, num_topics=10):
    grouped_text, topics = ldamodule.create_topics(text, sentence_group, num_topics)
    keywords = get_keywords(grouped_text, max_nodes)
    return [keywords, topics]

def get_mindmap(keywords, topics):
    fig, ax = plt.subplots(figsize=(40,25))
    G = nx.Graph()
    G.add_node('Mind Map')
    reg_exp_pattern = "[^\d\w\s]"
    for idx in topics:
        main_topic = re.sub(reg_exp_pattern, '', topics[idx]).strip()
        G.add_edge('Mind Map', main_topic)
        topic_keywords = keywords[idx]
        for keyword in topic_keywords:
            keyword = re.sub(reg_exp_pattern, '', keyword).strip()
            if(keyword != re.sub(reg_exp_pattern, '', topics[idx]).strip()):
                G.add_edge(main_topic, keyword)

    nx.draw(G, with_labels=True, node_size=2, font_size = 16, )
    return plt

def load_text(file_name_path, encoding="utf8"):
    with open(file_name_path, encoding=encoding) as f:
        input_text = f.readlines()
        f.close()
    final_text = ''
    for text in input_text:
        final_text += text
    final_text = clean_text(final_text)
    
    return final_text