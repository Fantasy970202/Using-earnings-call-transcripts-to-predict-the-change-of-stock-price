import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import re
import nltk
from wordcloud import WordCloud
from collections import Counter

imdb=pd.read_csv('complete_union.csv')

imdb['Day1'].value_counts().plot.pie(figsize=(6,6),title="Distribution of reviews per sentiment",labels=['',''],autopct='%1.1f%%')
labels=["Negative", "Positive"]
plt.legend(labels,loc=3)
plt.gca().set_aspect('equal')
plt.show()

imdb['Day5'].value_counts().plot.pie(figsize=(6,6),title="Distribution of reviews per sentiment",labels=['',''],autopct='%1.1f%%')
labels=["Negative", "Positive"]
plt.legend(labels,loc=3)
plt.gca().set_aspect('equal')
plt.show()

imdb.groupby('Day1').size().plot(kind='bar')
plt.show()
imdb.groupby('Day5').size().plot(kind='bar')
plt.show()

def generate_wordcloud(words,sentiment):
    plt.figure(figsize=(16,13))
    wc = WordCloud(background_color="white", max_words=100, max_font_size=50)
    wc.generate(words)
    plt.title("Most common {} words".format(sentiment), fontsize=20)
    plt.imshow(wc.recolor(colormap='Pastel2', random_state=17), alpha=0.98)
    plt.axis('off')
    plt.show()

print("Creating word clouds...")
positive_words=" ".join(imdb[imdb.Day1 == 1]['cleaned_transcript'].values)
negative_words=" ".join(imdb[imdb.Day1 == 0]['cleaned_transcript'].values)

generate_wordcloud(positive_words,"positive")
generate_wordcloud(negative_words,"negative")

