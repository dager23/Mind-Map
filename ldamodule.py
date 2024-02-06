import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.models import TfidfModel

def lemmatization(texts, stop_words ,allowed_postags=['NN', 'NNS', 'NNP', 'NNPS', 'RB', 'RBR', 'RBS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS']):
    '''
    Default Allowed postags are nouns, adjectives, verbs, adverb
    '''
    texts_out = []
    lemmatizer = WordNetLemmatizer()
    for text in texts:
        new_text = []
        words = nltk.word_tokenize(text)
        tagged_words = nltk.pos_tag(words)
        for tags in tagged_words:
            if((tags[0] not in stop_words) and (tags[1] in allowed_postags)):
                new_text.append(lemmatizer.lemmatize(tags[0]))
        final_text = " ".join(new_text)
        texts_out.append(final_text)
    
    return texts_out

def group_sentences(sentences, group_len = 3):
    new_sentences = []
    for idx in range(0, len(sentences), group_len):
        new_sent = ''
        i = idx
        while i<len(sentences) and i<idx+3:
            new_sent += sentences[i]
            i += 1
        new_sentences.append(new_sent)
    return new_sentences

def gen_words(texts):
    final = []
    for text in texts:
        new = gensim.utils.simple_preprocess(text, deacc=True) #Deacc is used to remove accents
        final.append(new)
    return final

def make_bigrams(bigram, texts):
    return [bigram[doc] for doc in texts]

def make_trigrams(trigram, bigram, texts):
    return [trigram[bigram[doc]] for doc in texts]

def get_topic_index(lda_model):
    cluster_idx = lda_model.show_topics()
    topics = {}
    present_topics = []
    for term in cluster_idx:
        terms = term[1].split('+')
        idx = 0
        element = terms[idx].split('*')[1]
        element = element.strip()[1:-1]
        while(element in present_topics):
            idx += 1
            element = terms[idx].split('*')[1]
            element = element.strip()[1:-1]
        present_topics.append(element)
        topics[term[0]] = element
        
    return topics

def get_clusted_sentences(lda_model, corpus, sentences):
    clustered_sentences = {}
    lda_corpus = lda_model[corpus]
    cluster_index_list = [doc for doc in lda_corpus]
    topics = get_topic_index(lda_model)
    for idx in range(0, len(cluster_index_list)):
        indexes = cluster_index_list[idx]
        if(len(indexes) == 1):
            clustered_sentences[sentences[idx]] = topics[indexes[0][0]]
        else:
            max_prob = 0
            topic = ''
            for index in indexes:
                prob = index[1]
                if(prob > max_prob):
                    max_prob = prob
                    topic = topics[index[0]]
            clustered_sentences[sentences[idx]] = topic
    return clustered_sentences

def get_grouped_sentences(lda_model, corpus, sentences):
    grouped_sentences = {k: '' for k in range(0, 10)}
    lda_corpus = lda_model[corpus]
    cluster_index_list = [doc for doc in lda_corpus]
    for idx in range(0, len(cluster_index_list)):
        indexes = cluster_index_list[idx]
        if(len(indexes) == 1):
            grouped_sentences[indexes[0][0]] += sentences[idx]
        else:
            max_prob = 0
            best_index = 0
            for index in indexes:
                prob = index[1]
                if(prob > max_prob):
                    max_prob = prob
                    best_index = index[0]
            grouped_sentences[best_index] += sentences[idx]

    return grouped_sentences
    
def create_topics(text, sentence_group=3, num_topics=10):
    stop_words = stopwords.words('english')
    sentences = nltk.sent_tokenize(text)

    grouped_sentences = group_sentences(sentences, sentence_group)
    lemmatized_text = lemmatization(grouped_sentences, stop_words)
    data_words = gen_words(lemmatized_text)

    bigram_phrases = gensim.models.Phrases(data_words, min_count=3, threshold=25)
    trigram_phrases = gensim.models.Phrases(bigram_phrases[data_words], threshold=25)

    bigram = gensim.models.phrases.Phraser(bigram_phrases)
    trigram = gensim.models.phrases.Phraser(trigram_phrases)

    data_bigrams = make_bigrams(bigram, data_words)
    data_bigrams_trigrams = make_trigrams(trigram, bigram, data_bigrams)

    id2word = corpora.Dictionary(data_bigrams_trigrams)
    texts = data_bigrams_trigrams
    corpus = [id2word.doc2bow(text) for text in texts] 

    tfidf = TfidfModel(corpus=corpus, id2word=id2word)
    low_value = 0.03
    words = []
    words_missing_in_tfid = []
    for i in range(0, len(corpus)):
        bow = corpus[i]
        low_value_words = []
        tfidf_ids = [id for id, value in tfidf[bow]]
        bow_ids = [id for id, value in bow]
        low_value_words = [id for id, value in tfidf[bow] if value < low_value]
        drops = low_value_words + words_missing_in_tfid
        for item in drops:
            words.append(id2word[item])
        words_missing_in_tfidf = [id for id in bow_ids if id not in tfidf_ids] # The words with tf-idf score 0 will be missing

        new_bow = [b for b in bow if b[0] not in low_value_words and b[0] not in words_missing_in_tfidf]     
        corpus[i] = new_bow

    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                               id2word=id2word,
                                               num_topics=num_topics,
                                               random_state=42,
                                               update_every=1,
                                               chunksize=100,
                                               passes=10,
                                               alpha='auto')
    topics = get_topic_index(lda_model)

    return [get_grouped_sentences(lda_model, corpus, grouped_sentences), topics]
