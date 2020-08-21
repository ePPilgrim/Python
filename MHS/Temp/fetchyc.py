import numpy as np
import pandas as pd
import os as os

TermsMap = {'1-Month' : 'M01', '2-Month' : 'M02', '3-Month' : 'M03', '4-Month' : 'M04', '5-Month' : 'M05', '6-Month' : 'M06',
        '7-Month' : 'M07', '8-Month' : 'M08', '9-Month' : 'M09', '10-Month' : 'M10', '11-Month' : 'M11', '12-Month' : 'M12',
        '1-Year' : 'Y01', '2-Year' : 'Y02', '3-Year' : 'Y03', '4-Year' : 'Y04', '5-Year' : 'Y05', '6-Year' : 'Y06',
        '7-Year' : 'Y07', '8-Year' : 'Y08', '9-Year' : 'Y09', '10-Year' : 'Y10', '12-Year' : 'Y12', '15-Year' : 'Y15', '20-Year' : 'Y20',
        '25-Year' : 'Y25', '30-Year' : 'Y30', '40-Year' : 'Y40', '50-Year' : 'Y50'}
CountriesMap = {'United Kingdom' : 'UK', 'Germany' : 'Germany', 'US' : 'US'}

def AppendNewYCs(df = pd.DataFrame(columns = list(TermsMap.values())), ycpath = r'.\data\raw\YC', countries = []):
    term_set = set(TermsMap.keys())
    for country in os.listdir(ycpath):
        if country in countries:
            dirpath = os.path.join(ycpath,country)
            subdf = pd.DataFrame(columns = ['Date'])
            if os.path.isdir(dirpath):
                concatMap = dict()
                for file in os.listdir(dirpath):
                    filepath = os.path.join(dirpath,file)
                    if os.path.isfile(filepath):
                        fparse = file.split()
                        term = term_set.intersection(set(fparse)).pop()
                        i = fparse.index(term)
                        state = ' '.join(fparse[:i])
                        temp_df = pd.read_csv(filepath, usecols = ['Date', 'Price'], converters = {'Date' : lambda x: pd.to_datetime(x)})
                        temp_df = temp_df.rename(columns = {'Price' : term})
                        if term in concatMap:
                            concatMap[term] = concatMap[term].append(temp_df, sort = True, ignore_index = True)
                        else:
                            concatMap[term] = temp_df
            for iterdf in concatMap.values():
                subdf = subdf.merge(iterdf,on = 'Date', sort = True, how = 'outer')
            subdf['Country'] = CountriesMap[country]
            subdf = subdf.rename(columns = TermsMap)
        df = df.append(subdf,sort = True, ignore_index = True)
    return df.drop_duplicates()

def GetUSBonds(usycpath):
    termsMap = {key.lower() : val for key, val in TermsMap.items()}
    temp_df = pd.read_csv(usycpath, converters =  {'Series Description' : lambda x: pd.to_datetime(x)})
    termsSet = set(termsMap.keys())
    colMap = {colName : termsSet.intersection(set(colName.split())) for colName in temp_df.columns}
    colMap['Series Description'] = set(['Date'])
    termsMap['Date'] = 'Date'
    colMap = {key : termsMap[val.pop()] for key, val in colMap.items()}
    temp_df['Country'] = 'US'
    temp_df = temp_df.replace('ND',np.nan)
    temp_df = temp_df.rename(columns = colMap)
    temp_df = pd.DataFrame(columns = list(TermsMap.values())).append(temp_df,sort = True, ignore_index = True)
    lx = temp_df[TermsMap.values()].isnull().all(axis = 1)
    return temp_df[~lx]
