import pandas as pd


#logic to get only the required columns from original survey file and save it (run only once)
'''
df = pd.read_csv("surveydata/survey.csv")
newdf = df.loc[:,['CASEID', 'YYYYMM', 'YYYY', 'ID', 'PAGO', 'HOM', 'GOVT', 'AGE', 'SEX', 'MARRY', 'REGION', 'EDUC']]
newdf.to_csv('surveydata/survey_subset.csv', index=False)

exit()
'''




Q1code = "GOVT" #"As the economic policy of the government ‐‐ I mean steps taken to fight inflation or unemployment ‐‐ would you say the government is doing a good job, only fair, or a poor job?
Q2code = "HOM" #"Generally speaking, do you think now is a good time or bad time to buy a house?"
Q3code =  "PAGO" #"We are interested in how people are getting along financially these days. Would you say that you (and your family living there) are better or worse off financially than you were ayear ago?"

Q1dict = {
    1: 5.0,
    3: 4.0,
    5: 3.0,
    8: 1.0,
    9: 0.0
}

Q2dict = {
    1: 5.0,
    3: 4.0,
    5: 3.0,
    8: 1.0,
    9: 0.0
}

Q3dict = {
    1: 5.0,
    3: 4.0,
    5: 3.0,
    8: 1.0,
    9: 0.0
}

ALL_DEMOGRAPHICS = ['AGE', 'SEX', 'MARRY', 'REGION', 'EDUC']

def get_name(demographics):
    fname = "surveydata/aggregated"
    for val in demographics:
        fname+="_"
        fname+=val
    return fname+".csv"


def generate_data(demographics = []):
    df = pd.read_csv("surveydata/survey_subset.csv")
 
    #transform Q1
    df['GOVT'] = df['GOVT'].apply(lambda x: Q1dict[x])
    df['HOM'] = df['HOM'].apply(lambda x: Q2dict[x])
    df['PAGO'] = df['PAGO'].apply(lambda x: Q3dict[x])
    
    groub_by_cols = ['YYYYMM']+demographics
    df_agg = df.groupby(groub_by_cols)[['GOVT', 'HOM', 'PAGO']].mean()
    return df_agg
    #df_agg.to_csv(get_name(demographics))


#generate_data(['SEX', 'MARRY'])
