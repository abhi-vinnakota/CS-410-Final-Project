# -*- coding: utf-8 -*-
"""CS 410 Final Project Data Pre-Processing.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/139u54FG2l4NpgEjVaFltOFPh4j9BG3XC

# **Part 1:** Download the Kaggle Dataset
"""

# Installs
!pip install opendatasets
!pip install pandas

# Imports
import ast
import csv
import numpy as np
import pandas as pd
import opendatasets as od
import matplotlib.pyplot as plt

from nltk.stem import PorterStemmer
from collections import defaultdict
from gensim.models import Word2Vec

ps = PorterStemmer()

# Download the dataset
od.download("https://www.kaggle.com/datasets/harshitshankhdhar/imdb-dataset-of-top-1000-movies-and-tv-shows")

# Create a new CSV file with a different delimiter
df = pd.read_csv("/content/imdb-dataset-of-top-1000-movies-and-tv-shows/imdb_top_1000.csv", sep=",")

new_rows = []
for i in range(len(df)):
  curr_row = []
  for val in df.iloc[i]:
    curr_row.append(val)
  new_rows.append(curr_row)

with open('/content/cleaned_movies.csv', 'w') as f:
  csvwriter = csv.writer(f)
  csvwriter.writerows(new_rows)

"""# **Part 2:** Hadoop Setup

## Install Java
"""

#install java
!apt-get install openjdk-8-jdk-headless -qq > /dev/null

#create java home variable
import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"

"""## Install Hadoop"""

#download HADOOP
!wget https://archive.apache.org/dist/hadoop/common/hadoop-3.3.0/hadoop-3.3.0.tar.gz

#decompress the Hadoop tar file
!tar -xzvf hadoop-3.3.0.tar.gz

#copy Hadoop directory to user/local
!cp -r hadoop-3.3.0/ /usr/local/

#find the default Java path
!readlink -f /usr/bin/java | sed "s:bin/java::"

#run Hadoop from /usr/local
!/usr/local/hadoop-3.3.0/bin/hadoop

#create input folder (test example)
!mkdir ~/testin

#copy example files to the input folder
!cp /usr/local/hadoop-3.3.0/etc/hadoop/*.xml ~/testin

#check that files have been successfully copied (10 files should appear)
!ls ~/testin

#run the mapreduce example (for sanity check)
!/usr/local/hadoop-3.3.0/bin/hadoop jar /usr/local/hadoop-3.3.0/share/hadoop/mapreduce/hadoop-mapreduce-examples-3.3.0.jar grep ~/testin ~/testout 'allowed[.]*'

#remove the testout content to reuse the folder for other excercises
!rm -r ~/testout

"""## Word Frequencies"""

#For you to do: Upload the files (left menu) for mapper and reducer.
#Then run these two lines to change their modes to execution
!chmod u+rwx /content/mapper.py
!chmod u+rwx /content/reducer.py

#Run hadoop to execute the mapper and reducer using the text.csv file and
# !/usr/local/hadoop-3.3.0/bin/hadoop jar /usr/local/hadoop-3.3.0/share/hadoop/tools/lib/hadoop-streaming-3.3.0.jar -input /content/train.csv -output ~/testout -file /content/mapper.py  -file /content/reducer.py  -mapper 'python mapper.py'  -reducer 'python reducer.py'

!/usr/local/hadoop-3.3.0/bin/hadoop jar /usr/local/hadoop-3.3.0/share/hadoop/tools/lib/hadoop-streaming-3.3.0.jar -input /content/imdb-dataset-of-top-1000-movies-and-tv-shows/imdb_top_1000.csv -output ~/testout -file /content/mapper.py  -file /content/reducer.py  -mapper 'python mapper.py'  -reducer 'python reducer.py'

#remove the output folder to run Hadoop again
!rm -r ~/testout

"""# **Part 3:** View Common Words in Descriptions"""

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

def plot_word_frequencies(word_frequencies):
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
    plt.show()

file_path = '/content/part-00000.txt'
# file_path = '/root/testout/part-00000'
word_frequencies = calculate_word_frequencies(file_path)
plot_word_frequencies(word_frequencies)

"""# **Part 4:** Obtain Top 10 Movies From Queries (Word2Vec)"""

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

# This is for the movie
csv_df = pd.read_csv('/content/cleaned_movies.csv', names=['Poster_Link', 'Series_Title', 'Released_Year', 'Certificate', 'Runtime', 'Genre', 'IMDB_Rating', 'Overview', 'Meta_score', 'Director', 'Star1', 'Star2', 'Star3', 'Star4', 'No_of_Votes', 'Gross'])
description_df = csv_df.get('Overview')
sorted_vocabulary, sorted_dict, cleaned_descr = create_vocabulary(description_df, 200)
print(cleaned_descr)

# Stem the queries (added query is "microsoft software technology")
queries = ["scary murder mystery", "young adult romance", "dystopian robot future", "documentary about war"]

# Use a skipgram model
w2v_model = Word2Vec(cleaned_descr, vector_size=300, window=2, sg=1, min_count=1)

# Find the scores using Word2Vec; get the average likelihood of one word in the query compared with every word in the document, then multiply the likelihoods of each query word
scores = [[], [], [], []]
for i in range(len(queries)):
  split_query = queries[i].split(" ")
  for j in range(len(cleaned_descr)):
    overall_score = 1
    for query_word in split_query:
      score, count = 0, 0
      for doc_word in cleaned_descr[j]:
        if doc_word not in w2v_model.wv or query_word not in w2v_model.wv:
          continue
        score += w2v_model.wv.similarity(query_word, doc_word)
        count += 1
      if count != 0: # Apparently there are some empty documents
        score /= count
      overall_score *= score
    scores[i].append(overall_score)

# For each query, get the top 10 most relevant documents (train data)
title_df = csv_df.get('Series_Title')
for i in range(len(queries)):
  print("[QUERY: \"" + queries[i] + "\"]")
  sorted_indices = np.flip(np.argsort(scores[i]))
  for j in range(10):
    print("Top Result " + str(j+1) + ": \"" + title_df[sorted_indices[j]] + "\" {Ranking/Relevance Score: " + str(scores[i][sorted_indices[j]]) + "}")

  if i != len(queries) - 1:
    print("-----------")

"""# **Part 5:** Obtain Top 10 Movies From Queries (Probabilistic Text Retrieval)"""

# Create a "dictionary" and "inverted index" from the output of MapReduce
word_dict = []
inverted_index = []
with open("/content/part-00000") as f:
  for line in f:
    line = line[:-1]
    word, metadata_list = line.split("	")
    metadata_list = ast.literal_eval(metadata_list)
    total_freq = 0
    for metadata in metadata_list:
      (docid, count) = tuple(metadata)
      total_freq += int(count)

      # Create the (term, docid, count) tuple for the inverted index
      inverted_index.append((word, docid, int(count)))

    # Create the (term, # of docs, total_freq) tuple for the dictionary
    word_dict.append((word, len(metadata_list), total_freq))

# Create a vocabulary using the top 200 most frequent words
word_dict.sort(key=lambda tup: tup[2], reverse=True)
vocab_dict = {}
for i in range(200):
  vocab_dict[word_dict[i][0]] = word_dict[i][2]
sorted_vocabulary = list(vocab_dict.keys())

print(sorted_vocabulary)
print(vocab_dict)

# This is for the movie
#csv_df = pd.read_csv('/content/imdb-dataset-of-top-1000-movies-and-tv-shows/imdb_top_1000.csv')
csv_df = pd.read_csv('/content/cleaned_movies.csv', names=['Poster_Link', 'Series_Title', 'Released_Year', 'Certificate', 'Runtime', 'Genre', 'IMDB_Rating', 'Overview', 'Meta_score', 'Director', 'Star1', 'Star2', 'Star3', 'Star4', 'No_of_Votes', 'Gross'])
num_rows = len(csv_df)

# Stem the queries (added query is "microsoft software technology")
queries = ["scary murder mystery", "young adult romance", "dystopian robot future", "documentary about war"]
stemmed_queries = []
for query in queries:
  stemmed_query = ""
  query_list = query.split(' ')
  for word in query_list:
    stemmed_query += (ps.stem(word) + " ")
  stemmed_query = stemmed_query.strip()
  stemmed_queries.append(stemmed_query)

# Create a count vector for each query
query_vectors = np.zeros((len(queries), len(sorted_vocabulary)))
for i in range(len(stemmed_queries)):
  split_query = stemmed_queries[i].split(" ")
  for word in split_query:
    if word in sorted_vocabulary:
      query_vectors[i][sorted_vocabulary.index(word)] += 1

# Create count vectors for each document using the top 200 words as the vocabulary; if the document contains a word not in the vocabulary, that word is ignored
document_vectors = np.zeros((num_rows, len(sorted_vocabulary)))
for i in range(len(inverted_index)):
  if inverted_index[i][0] in sorted_vocabulary:
    curr_index = sorted_vocabulary.index(inverted_index[i][0])
    document_vectors[inverted_index[i][1]-1][curr_index] = inverted_index[i][2]

# Create a data structure that maps each document to a document length
document_lens = {}
inverted_index.sort(key=lambda tup: tup[1])
for i in range(len(inverted_index)):
  if inverted_index[i][1] not in document_lens:
    document_lens[inverted_index[i][1]] = inverted_index[i][2]
  else:
    document_lens[inverted_index[i][1]] += inverted_index[i][2]

# Create a Collection LM containing probabilities of words from all documents
collection_lm = {}
total_words = 0
for i in range(len(word_dict)):
  total_words += word_dict[i][2]
for i in range(len(word_dict)):
  collection_lm[word_dict[i][0]] = word_dict[i][2] / total_words

# Obtain a probability of each word in the vocabulary using the Collection LM
vocab_lm = np.zeros(len(sorted_vocabulary))
for word in collection_lm:
  if word in sorted_vocabulary:
    vocab_lm[sorted_vocabulary.index(word)] = collection_lm[word]

# Find the score of each document by Probabilistic Ranking with JM-smoothing
lambda_val = 0.4
scores = [[], [], [], []]
for i in range(len(query_vectors)):
  for j in range(len(document_vectors)):
    second_half = None
    if document_lens[j+1] != 0:
      second_half = 1 + ((1-lambda_val)/lambda_val)*(document_vectors[j] / (document_lens[j+1]*vocab_lm))
    else:
      second_half = 1 + ((1-lambda_val)/lambda_val)*(document_vectors[j] / (vocab_lm))
    scores[i].append(np.dot(np.array(query_vectors[i]), np.log(second_half)))

# For the first 3 queries, get the top 10 most relevant documents (train data)
title_df = csv_df.get('Series_Title')
for i in range(4):
  print("[QUERY: \"" + queries[i] + "\"]")
  sorted_indices = np.flip(np.argsort(scores[i]))
  for j in range(10):
    print("Top Result " + str(j+1) + ": \"" + title_df[sorted_indices[j]] + "\" {Ranking/Relevance Score: " + str(scores[i][sorted_indices[j]]) + "}")

  if i != len(queries) - 1:
    print("-----------")

"""# **Part 5:** Apply Filters on Queries"""

# TODO: Maybe we need to compare Vector Space TF-IDF Text Retrieval with Probabilistic Text Retrieval (or word2vec; Charles can do this)
# TODO: We need to be able to filter by date and genre, copy the code above and modify it to account for different filters (this will have to utilize the CSV file)
# TODO: The UI needs to be created in order to account for everything that's been done so far
# TODO: Finish the report with proper figures

# For the first 3 queries, get the top 10 most relevant documents that fulfill the filter condition (train data)

# Should be able to filter by Year, Genre, Rating for now
title_df = csv_df.get('Series_Title')
genre_df = csv_df.get('Genre')
year_df = csv_df.get('Released_Year')
rating_df = csv_df.get('IMDB_Rating')

print(genre_df[1].split(', '))

filter_genre = True
filter_year_min = True
filter_year_max = True
filter_rating = True
desired_genres = ['Comedy']
desired_min_year = 2000
desired_max_year = 2010
desired_min_rating = 7
#check in the frontend to make sure that min <= max year
for i in range(1,2):
  print("[QUERY: \"" + queries[i] + "\"]")
  sorted_indices = np.flip(np.argsort(scores[i]))
  j = 0
  #make sure that we get 10 results, and that j doesn't go out of bounds
  results_so_far = 0
  while results_so_far < 10 and j < len(sorted_indices):
    fulfills_filters = True
    if filter_genre:
      genre_list = genre_df[sorted_indices[j]].split(', ')
      for k in range(len(desired_genres)):
        if desired_genres[k] not in genre_list:
          fulfills_filters = False
          break
    if fulfills_filters:
      if filter_year_min and int(year_df[sorted_indices[j]]) < desired_min_year:
        j += 1
        continue # same as fulfills_filters = False
      if filter_year_max and int(year_df[sorted_indices[j]]) > desired_max_year:
        j += 1
        continue
      if filter_rating and float(rating_df[sorted_indices[j]]) < desired_min_rating:
        j += 1
        continue
      print("Top Result " + str(results_so_far+1) + ": \"" + title_df[sorted_indices[j]] + "\" {Ranking/Relevance Score: " + str(scores[i][sorted_indices[j]]) + "}")
      print("Genres: ")
      print(genre_list)
      print("Year released: ")
      print(int(year_df[sorted_indices[j]]))
      print("Rating: ")
      print(float(rating_df[sorted_indices[j]]))
      results_so_far += 1
    j += 1
  if i != len(queries) - 1:
    print("-----------")