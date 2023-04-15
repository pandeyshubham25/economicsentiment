from getNews import getFilteredNews
from process import generate_data
from torch.utils.data import Dataset
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
import pickle

class NewsDataset(Dataset):

    #if idx is 0, it gives 2020-01 (indexes can be negative too, in that case we go to 2019)
    #count is the number of months to include starting from idx going up
    def get_month_strings(self, idx, count):
        base = "2020-01"
        dt = datetime.strptime(base + '-01', '%Y-%m-%d')
        
        res=[]
        for i in range(idx, idx+count):
            res.append( (dt + relativedelta(months=i)).strftime('%Y-%m'))
        
        return res

    #news_window = 1 means only current month is considered
    #picked_news: when you have the dictionary for a given date range pickled
    #pickle: True when you want to generate news dictionary and pickle it
    def __init__(self, demographics=[], 
                lemma=True, stemming = False, stopw = False, keywords=[], news_window = 1, metric = "GOVT",
                pickled_news_file=None, pickle_news=True, start="2019-01-01", end="2022-12-31"):
        #save imporant parameters
        self.demographics = demographics
        self.news_window = news_window
        self.metric = metric

        start = "2019-01-01"
        end = "2022-12-31"

        #prepare the survey data
        self.survey_df = generate_data(demographics)        

        #if news pickle file name has been provided
        if pickled_news_file is not None:
            with open(pickled_news_file, 'rb') as f:
                self.all_news=pickle.load(f)
        else:
            self.all_news = getFilteredNews(start, end, lemma=lemma, stemming = stemming, stopw = stopw, jsonFile = "data/newsAll.json", keywords = keywords)
            if pickle_news:
                with open("data/"+start+"_to_"+end+".pickle", 'wb') as f:
                    pickle.dump(self.all_news, f)
        
    
    def __len__(self):
        return len(self.survey_df)-12

    def __getitem__(self, idx):
        X = {}
        for demographic in self.demographics:
            X[demographic] = self.survey_df.iloc[idx][demographic]
        X["news"] = []

        news_st_idx = idx-self.news_window+1
        
        months = self.get_month_strings(news_st_idx, self.news_window)

        for month in months:
            X["news"]+=self.all_news[month]

        y = self.survey_df.iloc[idx][self.metric]
        return (X,y)

#teting code

'''
obj = NewsDataset(pickled_news_file="data/2019-01-01_to_2022-12-31.pickle", news_window=4)
print(obj.__len__())
#exit()
for i in range(obj.__len__()):
    X,y = obj.__getitem__(i)
    print(X["news"])
    exit()

'''


