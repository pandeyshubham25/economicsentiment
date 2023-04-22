from collections import defaultdict
import json
import pickle

import json
import nltk
import os
from datetime import datetime
from collections import defaultdict
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import re
def contains_keyword(s, keywords):
    s = s.lower()
    for keyword in keywords:
        if keyword.lower() in s:
            return True
    return False

def processTokens(sentence, lemma=True, stemming = False, stopw = False):

    tokens = word_tokenize(sentence)

    # Remove stop words
    if stopw:
        stop_words = set(stopwords.words('english'))
        tokens = [token.lower() for token in tokens if token.lower() not in stop_words]

    # Stem the tokens
    if stemming:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(token.lower()) for token in tokens]

    # Lemmatize the tokens
    if lemma:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens]


    return " ".join(tokens)    
def getFilteredNewsCap4(startDate, endDate, lemma=True, stemming = False, stopw = False, jsonFile = "data/newsAll.json", keywords = []):
    # print("################## starting to read news ##################")
    finalOutput = {}
    daynewsCount = defaultdict(int)
    with open(jsonFile, 'r') as f:
        data = json.load(f)
    
    # print("################## finished reading news ##################")
    for newsDate in data:
        if daynewsCount[newsDate] > 4:
            continue
        if startDate <= newsDate <= endDate:
            key = newsDate[:-3] ## it gets the year and month
            print(key)
            sentences = data[newsDate]
            for sentence in sentences:
                if daynewsCount[newsDate] > 4:
                    break
                if len(keywords) != 0: ### there are keywords that needs to be filtered
                    if contains_keyword(sentence, keywords):
                        sentence = processTokens(sentence, lemma=lemma, stemming = stemming, stopw = stopw)
                        if(key not in finalOutput):
                            finalOutput[key]=[]
                        finalOutput[key].append((sentence, newsDate))
                        daynewsCount[newsDate] += 1
                else: ### get every news that is in between starting and ending date
                    sentence = processTokens(sentence, lemma=lemma, stemming = stemming, stopw = stopw)
                    if(key not in finalOutput):
                        finalOutput[key]=[]
                    finalOutput[key].append((sentence, newsDate))
                    daynewsCount[newsDate] += 1
            print(daynewsCount[newsDate])
            print(finalOutput[key])
                
    return finalOutput

keywords_dict = {}
keywords_dict["GOVT"] = ["governemnt", "trump", "biden", "election", "president", "congress", "senate", "democrat", "republican", "political party", "political parties", "political ideology", "political beliefs", "political views", "political views", "political system", \
                        "political system", "hilary", "clinton", "bernie", "sanders", "democratic", "republican", "democratic party", "republican party", "democratic nominee", "republican nominee", "democrat", "republican", "dem"]
keywords_dict["HOM"] = ["home", "house", "mortgage", "rent", "interest rates", "housing market", "housing prices", "housing inventory", "housing affordability", "housing supply", "housing demand", "housing shortage", "housing crisis", "housing bubble", "housing crash"]
keywords_dict["PAGO"] = ["jobs", "inflation", "unemployment", "employment", "job market", "job growth", "job creation", "job loss", "job losses", "job openings", "job openings", "job openings", "job openings", "job openings", "job openings", "job openings", "job openings"]    

start = "2019-01-01"
end = "2022-05-31"

start_test = "2022-06-01"
end_test = "2022-12-31"
keywords = keywords_dict["GOVT"]
FilteredNewsCap4 = getFilteredNewsCap4(start, end, lemma=True, stemming = False, stopw = False, jsonFile = "data/newsAll.json", keywords = keywords)




with open("data/"+ start+"_to_"+ end +"_" + keywords[0] + "_train" + "_cap4" + ".pickle", 'wb') as f:
        pickle.dump(FilteredNewsCap4, f)

FilteredNewsCap4 = getFilteredNewsCap4(start_test, end_test, lemma=True, stemming = False, stopw = False, jsonFile = "data/newsAll.json", keywords = keywords)

with open("data/"+ start_test+"_to_"+ end_test +"_" + keywords[0] + "_test" + "_cap4" + ".pickle", 'wb') as f:
        pickle.dump(FilteredNewsCap4, f)
# keywords = keywords_dict["HOM"]
# keywords = keywords_dict["PAGO"]


keywords = keywords_dict["HOM"]

with open("data/"+ start+"_to_"+ end +"_" + keywords[0] + "_train" + "_cap4" + ".pickle", 'wb') as f:
        pickle.dump(FilteredNewsCap4, f)

FilteredNewsCap4 = getFilteredNewsCap4(start_test, end_test, lemma=True, stemming = False, stopw = False, jsonFile = "data/newsAll.json", keywords = keywords)

with open("data/"+ start_test+"_to_"+ end_test +"_" + keywords[0] + "_test" + "_cap4" + ".pickle", 'wb') as f:
        pickle.dump(FilteredNewsCap4, f)


keywords = keywords_dict["PAGO"]

with open("data/"+ start+"_to_"+ end +"_" + keywords[0] + "_train" + "_cap4" + ".pickle", 'wb') as f:
        pickle.dump(FilteredNewsCap4, f)

FilteredNewsCap4 = getFilteredNewsCap4(start_test, end_test, lemma=True, stemming = False, stopw = False, jsonFile = "data/newsAll.json", keywords = keywords)

with open("data/"+ start_test+"_to_"+ end_test +"_" + keywords[0] + "_test" + "_cap4" + ".pickle", 'wb') as f:
        pickle.dump(FilteredNewsCap4, f)