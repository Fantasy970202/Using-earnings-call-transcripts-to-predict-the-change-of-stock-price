Prerequisite: download GoogleNews-vectors-negative300.bin FROM  https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing
python 3.11

Files explanation:
1, earning_call: Six companies earning calls from 2018-2022.

2, stock_price: Six companies' stock price changes from 2018-2022.

3, NASDAQ_HistoricalData.csv: NASDAQ market data change from 2017-2022.

4, NDAQ_csv_clean.py: Delete unrelated columns and change the 'Date' format to suit other csv files' 'Date'.
And storing to NASDAQ_17_22.csv.

5, stock_label_txtName.py: Calculating 'Day1' and 'Day2' labels and combine them with Company tickers,transcripts releasing Date and Transcript_path.
And storing to trans_stock_union.csv.

6, transcripts_clean.py: Open trans_stock_union.csv, use its 'Date' to lock the path of relative earning call transcripts.
Then, clean the transcripts' txt data by using NLTK. Finally, union the labels with cleaned transcripts storing to 'complete_union.csv'.

7, TFIDF_day1.py and TFIDF_day5.py: Simply run the files, and it will output four models' results.
Day1 means the results by using label 'Day1', day5 means the results by using label 'Day5'

8, word2vec_day1.py and word2vec_day5.py: Simply run the files, and it will output seven models' results.
Day1 means the results by using label 'Day1', day5 means the results by using label 'Day5'.
And it will output the predicting results by using its inside function called " FunctionPredictUrgency"

9, visualization.py: simply run the code.

Additional:
Since I have already finished data cleaning, we don't have to run 4, NDAQ_csv_clean.py, 5, stock_label_txtName.py, 6, transcripts_clean.py.
If you want to check those code, just run those py file by order of: 1, NDAQ_csv_clean.py, 2, stock_label_txtName.py, 3, transcripts_clean.py

Contribution:
Xiongxiang Fan(xf634):Nearly all of the coding. 70% of data collecting.
1, stock_price csv data collecting, NASDAQ_HistoricalData csv data collecting.
2, stock price data cleaning, generating 'Day1' and 'Day5' labels' results.
3, all of the earning calls' texts cleaning.
4, union stock price labels with cleaning transcripts storing as 'complete_union.csv'.
5, using word cloud, pie chart and bar chart for visualizing
6, TF-IDF, word2vec vectorizing
7, TF-IDF-SVM, TF-IDF-LogisticRegression, TF-IDF-Random_Forest and TF-IDF-XGBoost modeling.
8, word2vec-LogisticRegression, word2vec-Naive_Bayes, word2vec-KNN, word2vec-Logistic_Regression and word2vec-Adaboost modeling.
9, new text data predicting function
10, README.txt