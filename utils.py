from collections import defaultdict
import ast
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from nltk.stem import PorterStemmer
from gensim.models import Word2Vec

def calculate_word_frequencies(file_path):
    word_freq = defaultdict(int)

    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                word = parts[0]
                occurrences = ast.literal_eval(parts[1])

                for _, count in occurrences:
                    word_freq[word] += count

    return word_freq

def plot_word_frequencies(word_frequencies, filepath='static/word_frequencies.png'):
    sorted_word_frequencies = sorted(word_frequencies.items(), key=lambda x: x[1], reverse=True)

    top_words = sorted_word_frequencies[:50]

    words, frequencies = zip(*top_words)

    plt.figure(figsize=(12, 8))
    plt.bar(words, frequencies, color='skyblue')
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.title('Top 50 Word Frequencies')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

"""
Returns a list of words (vocabulary) sorted in descending order of frequency, a dictionary mapping of those words to their frequencies, and a list containing cleaned descriptions for each document.
"""
def create_vocabulary(dataframe, n_frequent):
  numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
  symbols = ['!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '-', '=', '_', '+', '{', '}', '|', '[', ']', '\\', ':', '"', ';', '\'', '<', '>', '?', ',', '.', '/']
  stop_words = ['a', 'an', 'and', 'are', 'as', 'at', 'be', 'but', 'by', 'for', 'if', 'in', 'into', 'is', 'it', 'no', 'not', 'of', 'on', 'or', 'such', 'that',
                'the', 'their', 'then', 'there', 'these', 'they', 'this', 'to', 'was', 'will', 'with']

  word_count_dict = {}
  temp_df = dataframe.copy()
  cleaned_descriptions = []
  for i in range(len(temp_df)):

    temp_df.iloc[i] = temp_df.iloc[i].lower()

    lt_i = temp_df.iloc[i].find('&lt')
    if lt_i != -1:
      temp_df.iloc[i] = temp_df.iloc[i][0:lt_i]

    for number in numbers:
      if number in temp_df.iloc[i]:
        temp_df.iloc[i] = temp_df.iloc[i].replace(number, ' ')
    for symbol in symbols:
      if symbol in temp_df.iloc[i]:
        temp_df.iloc[i] = temp_df.iloc[i].replace(symbol, ' ')

    split_line = temp_df.iloc[i].split(' ')
    clean_line = []
    for j in range(len(split_line)):
      curr_word = split_line[j]
      if len(curr_word) != 0 and curr_word not in stop_words and len(curr_word) != 1:
        clean_line.append(curr_word)
    cleaned_descriptions.append(clean_line)

    for j in range(len(clean_line)):
      curr_word = clean_line[j]
      if curr_word not in word_count_dict:
        word_count_dict[curr_word] = 1
      else:
        word_count_dict[curr_word] += 1

  keys, values= list(word_count_dict.keys()), list(word_count_dict.values())
  sorted_value_i = np.flip(np.argsort(values))
  sorted_dict = {keys[i]: values[i] for i in sorted_value_i}
  top_n_words = list(sorted_dict.keys())[0:n_frequent]

  return top_n_words, sorted_dict, cleaned_descriptions

def process_query(query, cleaned_descr, w2v_model, csv_df, genre, min_year, max_year, min_rating):
    
    split_query = query.split(" ")
    scores = []
    for j in range(len(cleaned_descr)):
        overall_score = 1
        for query_word in split_query:
            score, count = 0, 0
            for doc_word in cleaned_descr[j]:
                if doc_word not in w2v_model.wv or query_word not in w2v_model.wv:
                    continue
                score += w2v_model.wv.similarity(query_word, doc_word)
                count += 1
            if count != 0: 
                score /= count
            overall_score *= score
        scores.append(overall_score)

    sorted_indices = np.flip(np.argsort(scores))
    filtered_movies = [{"title": csv_df.iloc[index]['Series_Title'],
                    "poster": csv_df.iloc[index]['Poster_Link'],
                    "year": int(csv_df.iloc[index]['Released_Year']),
                    "genre": csv_df.iloc[index]['Genre'],
                    "imdb": csv_df.iloc[index]["IMDB_Rating"],
                    "overview": csv_df.iloc[index]["Overview"],
                    "score": scores[index]}
                   for index in sorted_indices
                   if (genre in csv_df.iloc[index]['Genre']) and
                      (csv_df.iloc[index]['Released_Year'] >= int(min_year)) and
                      (csv_df.iloc[index]['Released_Year'] <= int(max_year)) and
                      (float(csv_df.iloc[index]['IMDB_Rating']) >= float(min_rating))]

    top_movies = filtered_movies[:10] if len(filtered_movies) >= 10 else filtered_movies

    return top_movies

def process_query_PRM(query, word_dict, inverted_index, csv_df, genre, min_year, max_year, min_rating):
    vocab_dict = {}
    for i in range(200):
        vocab_dict[word_dict[i][0]] = word_dict[i][2]
    
    sorted_vocabulary = list(vocab_dict.keys())

    ps = PorterStemmer()
    stemmed_query = " ".join(ps.stem(word) for word in query.split())
    
    query_vector = np.zeros(len(sorted_vocabulary))
    for word in stemmed_query.split():
        if word in sorted_vocabulary:
            query_vector[sorted_vocabulary.index(word)] += 1

    num_rows = len(csv_df)
    document_vectors = np.zeros((num_rows, len(sorted_vocabulary)))
    for i in range(len(inverted_index)):
        if inverted_index[i][0] in sorted_vocabulary:
            curr_index = sorted_vocabulary.index(inverted_index[i][0])
            document_vectors[inverted_index[i][1]-1][curr_index] = inverted_index[i][2]

    document_lens = {}
    inverted_index.sort(key=lambda tup: tup[1])
    for i in range(len(inverted_index)):
        if inverted_index[i][1] not in document_lens:
            document_lens[inverted_index[i][1]] = inverted_index[i][2]
        else:
            document_lens[inverted_index[i][1]] += inverted_index[i][2]

    collection_lm = {}
    total_words = 0
    for i in range(len(word_dict)):
        total_words += word_dict[i][2]
        for i in range(len(word_dict)):
            collection_lm[word_dict[i][0]] = word_dict[i][2] / total_words

    vocab_lm = np.zeros(len(sorted_vocabulary))
    for word in collection_lm:
        if word in sorted_vocabulary:
            vocab_lm[sorted_vocabulary.index(word)] = collection_lm[word]
    
    lambda_val = 0.4
    scores = []
    for j in range(len(document_vectors)):
        second_half = None
        if document_lens[j+1] != 0:
            second_half = 1 + ((1-lambda_val)/lambda_val)*(document_vectors[j] / (document_lens[j+1]*vocab_lm))
        else:
            second_half = 1 + ((1-lambda_val)/lambda_val)*(document_vectors[j] / (vocab_lm))
        scores.append(np.dot(query_vector, np.log(second_half)))

    sorted_indices = np.flip(np.argsort(scores))

    filtered_movies = [{"title": csv_df.iloc[index]['Series_Title'],
                    "poster": csv_df.iloc[index]['Poster_Link'],
                    "year": int(csv_df.iloc[index]['Released_Year']),
                    "genre": csv_df.iloc[index]['Genre'],
                    "imdb": csv_df.iloc[index]["IMDB_Rating"],
                    "overview": csv_df.iloc[index]["Overview"],
                    "score": scores[index]}
                   for index in sorted_indices
                   if (genre in csv_df.iloc[index]['Genre']) and
                      (csv_df.iloc[index]['Released_Year'] >= int(min_year)) and
                      (csv_df.iloc[index]['Released_Year'] <= int(max_year)) and
                      (float(csv_df.iloc[index]['IMDB_Rating']) >= float(min_rating))]

    top_movies = filtered_movies[:10] if len(filtered_movies) >= 10 else filtered_movies
    
    return top_movies