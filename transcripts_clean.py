import nltk
import eli5
from matplotlib import pyplot as plt
import pandas as pd
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
def clean(ticker, path):
    file_path = 'earning_call/' + ticker + '/' + path
    with open(file_path, 'r') as f:
        data = f.read()
        data2 = data.replace('’','')
    #remove stopwords:
        lowered_token = list(map(str.lower, word_tokenize(data2)))
    #print(lowered_token)
        stpwrd = nltk.corpus.stopwords.words('english')
        stpwrd.extend(["aapl","apple","amazon","amgen","costco", "microsoft","micron","nasdaq"])
        stopwords_en = set(stpwrd)
    #remove punctuation
        stopwords_en_punct = stopwords_en.union(set(punctuation))
        x = []
        for word in lowered_token:
            if word not in stopwords_en_punct:
                if word.isalpha():
                    x.append(word)

    #Lemmatizing
        stop_pun_stem_lem = []
        wnl = WordNetLemmatizer()
        for word in x:
            stop_pun_stem_lem.append(wnl.lemmatize(word))
    #print(stop_pun_stem_lem)
        return ' '.join(stop_pun_stem_lem)
        #wordcloud = WordCloud().generate(' '.join(stop_pun_stem_lem))
        #plt.imshow(wordcloud, interpolation='bilinear')
        #plt.axis("off")
        #plt.savefig('wordcloud11.png')
        #plt.show()
        #fd = nltk.FreqDist(stop_pun_stem_lem)
        #print(fd.most_common(3))
        #eli5.show_weights(fd.most_common(3))
if __name__ == '__main__':
    df = pd.read_csv('trans_stock_union.csv')
    for i in range(len(df)):
        df.loc[i, 'cleaned_transcript'] = clean(df.iloc[i]['Company'], df.iloc[i]['Transcript_path'])
    # print(df.head())
    df.drop(['Unnamed: 0'], axis=1, inplace=True)
    #print(df)
    df.to_csv('complete_union.csv', index="False")
