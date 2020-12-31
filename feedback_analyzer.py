# Required Libraries
import os
import sys
import re
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import csv
import time
from datetime import datetime
from zipfile import ZipFile
from pyspark import SparkConf, SparkContext
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from statistics import mean
from collections import Counter
from wordcloud import WordCloud, STOPWORDS
from PIL import Image

'''Input'''

print("Welcome to Feedback Analyzer!")

print("Instructions:")
print("------------------------------------------------------------------------------------------")
print("1. Ensure your input file is a CSV FIle.")
print("2. Ensure your column having feedbacks should be named 'Feedback'.")
print("3. Ensure your input file exists in the same directory where you'll be executing the code.")
print("------------------------------------------------------------------------------------------")
input_file = sys.argv[1]
print(input_file,"been used.")


'''Word Count'''

print("Word Count Started")

# Clean Feedbacks
def normalizeWords(text):
    return re.compile(r'\W+', re.UNICODE).split(text.lower())

# Spark Instance
conf = SparkConf().setMaster("local").setAppName("WordCount")
sc = SparkContext(conf = conf)

# Files for counting words
csv_file = 'filtered_feedback.csv'
txt_file = 'result.txt'

# Extract Feedback column
data = pd.read_csv(input_file, index_col ="Feedback" ) 
data = data.loc[:, data.columns.intersection(['Feedback'])]
data.to_csv('filtered_feedback.csv', header=False) 

# Encoding Feedback
with open(txt_file, "w") as my_output_file:
    with open(csv_file, "r", encoding='cp1252',  errors="ignore") as my_input_file:
        [ my_output_file.write(" ".join(row)+'\n') for row in csv.reader(my_input_file)]
    my_output_file.close()

# Spark Counting Words
input = sc.textFile("file:///Feedback_Analyzer/result.txt")
words = input.flatMap(normalizeWords)

wordCounts = words.map(lambda x: (x, 1)).reduceByKey(lambda x, y: x + y)
wordCountsSorted = wordCounts.map(lambda x: (x[1], x[0])).sortByKey()
results = wordCountsSorted.collect()

word_count_result = {}
for result in results:
    count = str(result[0])
    word = result[1].encode('ascii', 'ignore')
    if (word):
        word_count_result[str(word.decode())] = [int(count)]
        '''print(word.decode() + ":\t\t" + count)'''

# CSV file for result of word count
dataframe = pd.DataFrame.from_dict(word_count_result, orient='index')
dataframe.to_csv('word_count_result.csv',header=False) 

print("Word Count Completed")
print("------------------------------------------------------")


'''Sentiment Analysis'''

print("Sentiment Analysis Started")
print("Sit back and relax!! This might take several minutes.")

# Remove Emojis from feedback
def remove_emoji(string):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)

# Textblob Evaluation 
def textblob_evaluation(polarity):
    if polarity >= 0.05: return 1
    elif polarity <= -0.05: return -1
    else: return 0

# Vader Evaluation 
def vader_evaluation(polarity):
    if polarity >= 0.1: return 1
    elif polarity <= -0.4: return -1
    else: return 0

# Algorithm for Sentiment Analysis
def sentiment_evaluation(polarities):
    polarity_counter = Counter(polarities)
    pos,neu,neg = polarity_counter[1],polarity_counter[0],polarity_counter[-1] 
    polarity_list = { "-1" : neg, "0" : neu, "1" : pos }
    # print(polarity_list)
    if pos == neu == neg == 0: return 0 
    if neg > 1 and (neg >= pos-1 or neg >= neu-1): return -1
    if pos >= neu and pos > neg: return 1
    return max(polarity_list, key= lambda x: polarity_list[x])

# Create Clean Text
def clean_text(line):
    f = open("clean_text.txt", "a")
    f.write(line+"\n")
    f.close()

# Analyze Sentiment for each Feedback
def analyze_sentiment(line):
    polarities = []
    sid_obj = SentimentIntensityAnalyzer()
    line = remove_emoji(str(line))
    line = (str(line).encode('ascii', 'ignore')).decode()
    clean_text(line)
    for sentence in line.split('.'):
        if len(sentence)>0:
            # print(sentence)
            textblob_result,vader_result = TextBlob(str(sentence)).sentiment.polarity, sid_obj.polarity_scores(sentence)['compound'] 
            textblob_eval, vader_eval = textblob_evaluation(textblob_result), vader_evaluation(vader_result)
            # print(textblob_result,vader_result)
            if(textblob_eval == -1 or vader_eval == -1): polarities.append(-1)
            elif(textblob_eval == 0 and vader_eval == 0): polarities.append(0)
            elif(textblob_eval == 0 or vader_eval == 0):
                if(vader_eval != 0) : polarities.append(vader_eval)
                else: polarities.append(0)
            elif(textblob_eval == 1 and vader_eval == 1): polarities.append(1)    
    result = int(sentiment_evaluation(polarities))
    if result>0: return "Positive"
    if result==0: return "Neutral"
    if result<0: return "Negative"

# Generate file after Sentiment analysis
df = pd.read_csv(input_file)

sentiment = np.array([])
for feedback in df["Feedback"]:
    sentiment = np.append(sentiment, np.array([analyze_sentiment(feedback)]))
df["Sentiment"] = sentiment
df = df.loc[:, df.columns.intersection(['Feedback','Sentiment'])]
df.index += 1
df.to_csv('sentiment_analysis.csv', header=False)

# Generate WordCloud
text = open('clean_text.txt', 'r').read()
stopwords = STOPWORDS
mask = np.array(Image.open("cloud.png"))
wc = WordCloud(stopwords = stopwords, mask=mask, background_color="white")
wc.generate(text)
wc.to_file('wordcloud.png')

print("Sentiment Analysis Completed")
print("------------------------------------------------------")


'''Zip File Generation'''

print("Zipping Files")

# Current Timestamp
dt_object = datetime.fromtimestamp(time.time())

# Process details
f = open("details.txt", "w")
details = "Timestamp: "+str(dt_object)+"\nInput File: "+str(input_file)
f.write(str(details))
f.close()

# Zipping Files
filename = "fa_"+str(round(time.time()))+".zip"
zipObj = ZipFile(str(filename), "w")
zipObj.write('details.txt')
zipObj.write('sentiment_analysis.csv')
zipObj.write('word_count_result.csv')
zipObj.write('wordcloud.png')
zipObj.close()

# Remove Files
scrap_files = [
    "details.txt",
    "filtered_feedback.csv",
    "result.txt",
    "sentiment_analysis.csv",
    "word_count_result.csv",
    "clean_text.txt",
    "wordcloud.png"
]
for scrap_file in scrap_files:
    try: 
        os.remove(scrap_file) 
        # print("% s removed successfully" % scrap_file) 
    except OSError as error: 
        print(error) 
        print("File path %s can not be removed" % scrap_file) 

print("Your required ZIP file "+str(filename)+" is ready.")
print("------------------------------------------------------")
