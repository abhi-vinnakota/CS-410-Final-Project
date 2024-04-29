from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from utils import calculate_word_frequencies, plot_word_frequencies, create_vocabulary, process_query, process_query_PRM
import sys
import ast

app = Flask(__name__)

csv_df = pd.read_csv('./cleaned_movies.csv', names=['Poster_Link', 'Series_Title', 'Released_Year', 'Certificate', 'Runtime', 'Genre', 'IMDB_Rating', 'Overview', 'Meta_score', 'Director', 'Star1', 'Star2', 'Star3', 'Star4', 'No_of_Votes', 'Gross'])

description_df = csv_df.get('Overview')

sorted_vocabulary, sorted_dict, cleaned_descr = create_vocabulary(description_df, 200)

w2v_model = Word2Vec(cleaned_descr, vector_size=300, window=2, sg=1, min_count=1)

csv_df['Released_Year'] = pd.to_numeric(csv_df['Released_Year'], errors='coerce') 

word_dict = []
inverted_index = []
with open("./part-00000") as f:
  for line in f:
    line = line[:-1]
    word, metadata_list = line.split("	")
    metadata_list = ast.literal_eval(metadata_list)
    total_freq = 0
    for metadata in metadata_list:
      (docid, count) = tuple(metadata)
      total_freq += int(count)

      inverted_index.append((word, docid, int(count)))

    word_dict.append((word, len(metadata_list), total_freq))

word_dict.sort(key=lambda tup: tup[2], reverse=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        genre = request.form.get('genre', '')
        min_year = int(request.form.get('min_year', 0))
        max_year = int(request.form.get('max_year', 3000))
        min_rating = float(request.form.get('min_rating', 0))

        filtered_df = csv_df[
            (csv_df['Genre'].str.contains(genre)) &
            (csv_df['Released_Year'] >= min_year) &
            (csv_df['Released_Year'] <= max_year) &
            (csv_df['IMDB_Rating'] >= min_rating)
        ]
        data = [('Filtered Movies', filtered_df.to_html(classes='data'))]

        return render_template('results.html', data=data)
    else:
        return render_template('index.html')
    
@app.route('/plot-frequencies')
def plot_frequencies():
    file_path = 'part-00000'  # Ensure this path is correct
    word_frequencies = calculate_word_frequencies(file_path)
    plot_word_frequencies(word_frequencies)  # This will save the image to 'static/word_frequencies.png'
    return send_from_directory(directory='static', path='word_frequencies.png')

@app.route('/movie-query', methods=['GET', 'POST'])
def movie_query():
    if request.method == 'POST':
        query = request.form['query']
        genre = request.form['genre'] if request.form['genre'] != '' else ''
        min_year = request.form['min_year'] if request.form['min_year'] != '' and request.form['min_year'] != None else 0
        max_year = request.form['max_year'] if request.form['max_year'] != '' and request.form['max_year'] != None else 3000
        min_rating = request.form['min_rating'] if request.form['min_rating'] != '' and request.form['min_rating'] != None else 0
        
        top_movies = process_query(query, cleaned_descr, w2v_model, csv_df, genre, min_year, max_year, min_rating)
        return render_template('results.html', top_movies=top_movies)
    return render_template('index.html')

@app.route('/movie-query-prm', methods=['GET', 'POST'])
def movie_query_prm():
    if request.method == 'POST':
        query = request.form['query']
        genre = request.form['genre'] if request.form['genre'] != '' else ''
        min_year = request.form['min_year'] if request.form['min_year'] != '' and request.form['min_year'] != None else 0
        max_year = request.form['max_year'] if request.form['max_year'] != '' and request.form['max_year'] != None else 3000
        min_rating = request.form['min_rating'] if request.form['min_rating'] != '' and request.form['min_rating'] != None else 0
        top_movies = process_query_PRM(query, word_dict, inverted_index, csv_df, genre, min_year, max_year, min_rating)
        return render_template('results.html', top_movies=top_movies)
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
