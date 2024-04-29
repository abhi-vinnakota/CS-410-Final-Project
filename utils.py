from collections import defaultdict
import ast
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Plot top 50 words and their respective frequencies
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
  # Lists of characters/words to remove
  numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
  symbols = ['!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '-', '=', '_', '+', '{', '}', '|', '[', ']', '\\', ':', '"', ';', '\'', '<', '>', '?', ',', '.', '/']
  stop_words = ['a', 'an', 'and', 'are', 'as', 'at', 'be', 'but', 'by', 'for', 'if', 'in', 'into', 'is', 'it', 'no', 'not', 'of', 'on', 'or', 'such', 'that',
                'the', 'their', 'then', 'there', 'these', 'they', 'this', 'to', 'was', 'will', 'with']

  word_count_dict = {}
  temp_df = dataframe.copy()
  cleaned_descriptions = []
  for i in range(len(temp_df)):
    # Set all words to lowercase
    temp_df.iloc[i] = temp_df.iloc[i].lower()

    # Remove weird links
    lt_i = temp_df.iloc[i].find('&lt')
    if lt_i != -1:
      temp_df.iloc[i] = temp_df.iloc[i][0:lt_i]

    # Remove numbers and punctuation (symbols)
    for number in numbers:
      if number in temp_df.iloc[i]:
        temp_df.iloc[i] = temp_df.iloc[i].replace(number, ' ')
    for symbol in symbols:
      if symbol in temp_df.iloc[i]:
        temp_df.iloc[i] = temp_df.iloc[i].replace(symbol, ' ')

    # Remove whitespace from words and remove stop-words
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

def process_query(query, cleaned_descr, w2v_model, csv_df):
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
            if count != 0:  # Handle empty documents
                score /= count
            overall_score *= score
        scores.append(overall_score)

    sorted_indices = np.flip(np.argsort(scores))
    top_movies = [{"title": csv_df.iloc[index]['Series_Title'], "score": scores[index]} for index in sorted_indices[:10]]

    return top_movies
  
def PRM_process_query(query, sorted_vocabulary, filtered_df):
    query_terms = query.lower().split()
    movie_scores = []
    for index, row in filtered_df.iterrows():
        movie_score = 0
        for term in query_terms:
            if term in sorted_vocabulary:
                term_index = sorted_vocabulary.index(term)
                term_frequency = row['Overview'].lower().count(term)
                doc_length = len(row['Overview'].split())
                avg_doc_length = np.mean([len(doc.split()) for doc in filtered_df['Overview']])
                collection_frequency = sum([doc.lower().count(term) for doc in filtered_df['Overview']])
                prm_score = (1 + np.log(1 + np.log(term_frequency))) / ((1 - 0.5) + 0.5 * (doc_length / avg_doc_length)) * np.log((len(filtered_df) - collection_frequency + 0.5) / (collection_frequency + 0.5))
                movie_score += prm_score
        movie_scores.append((row['Series_Title'], movie_score))
    movie_scores.sort(key=lambda x: x[1], reverse=True)
    top_movies = [{'Title': title, 'Score': score} for title, score in movie_scores[:10]]
    return top_movies
