import numpy as np
import pandas as pd
import os as os
import tensorflow as tf

TermMapToMonth = { 'M01' : 1.0, 'M02' : 2.0, 'M03' : 3.0, 'M04' : 4.0, 'M05' : 5.0, 'M06' : 6.0, 'M07' : 7.0, 'M08' : 8.0,
                    'M09' : 9.0, 'M10' : 10.0, 'M11' : 11.0, 
                    'Y01' : 12.0, 'Y02' : 24.0, 'Y03' : 36.0, 'Y04' : 48.0, 'Y05' : 60.0, 'Y06' : 72.0, 'Y07' : 84.0,
                    'Y08' : 96.0, 'Y09' : 108.0, 'Y10' : 120.0, 'Y12' : 144.0, 'Y15' : 180.0, 'Y20' : 240.0,
                    'Y25' : 300.0, 'Y30' : 360.0, 'Y40' : 480.0, 'Y50' : 600.0}

TermsMap = {'1-Month' : 'M01', '2-Month' : 'M02', '3-Month' : 'M03', '4-Month' : 'M04', '5-Month' : 'M05', '6-Month' : 'M06',
        '7-Month' : 'M07', '8-Month' : 'M08', '9-Month' : 'M09', '10-Month' : 'M10', '11-Month' : 'M11',
        '1-Year' : 'Y01', '2-Year' : 'Y02', '3-Year' : 'Y03', '4-Year' : 'Y04', '5-Year' : 'Y05', '6-Year' : 'Y06',
        '7-Year' : 'Y07', '8-Year' : 'Y08', '9-Year' : 'Y09', '10-Year' : 'Y10', '12-Year' : 'Y12', '15-Year' : 'Y15', '20-Year' : 'Y20',
        '25-Year' : 'Y25', '30-Year' : 'Y30', '40-Year' : 'Y40', '50-Year' : 'Y50'}
CountriesMap = {'United Kingdom' : 'UK', 'USA' : 'US'}

def create_df_usa(countrypath):
    termsMap = {key.lower() : val for key, val in TermsMap.items()}
    for file in os.listdir(countrypath):
        filepath = os.path.join(countrypath,file)
        if os.path.isfile(filepath): 
            print(filepath)
            temp_df = pd.read_csv(filepath, converters =  {'Series Description' : lambda x: pd.to_datetime(x)})
            termsSet = set(termsMap.keys())
            colMap = {colName : termsSet.intersection(set(colName.split())) for colName in temp_df.columns}
            colMap['Series Description'] = set(['Date'])
            termsMap['Date'] = 'Date'
            colMap = {key : termsMap[val.pop()] for key, val in colMap.items()}
            temp_df = temp_df.replace('ND',np.nan)
            temp_df = temp_df.rename(columns = colMap)
            temp_df = pd.DataFrame(columns = list(TermsMap.values())).append(temp_df,sort = True, ignore_index = True)
            lx = temp_df[TermsMap.values()].isnull().all(axis = 1)
            return temp_df[~lx]

def create_df_other(countrypath):
    term_set = set(TermsMap.keys()) 
    df = pd.DataFrame(columns = ['Date'])
    concatMap = dict()
    for file in os.listdir(countrypath):
        filepath = os.path.join(countrypath,file)
        print(filepath)
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
        df = df.merge(iterdf,on = 'Date', sort = True, how = 'outer')
    return df.rename(columns = TermsMap)

def save_df(df, destdir, country, continent):
    filename = '{}_{}_{}.csv'.format(continent, country, df.shape[0])
    filepath = os.path.join(destdir,filename)
    return df.to_csv(filepath,index=False)

def FirstConversion(srcdir = r'.\data\raw\YC', destdir = r'.' , asnan = -256.0):   
    for continent in os.listdir(srcdir):
        continentpath = os.path.join(srcdir,continent)
        for country in os.listdir(continentpath):
            countrypath = os.path.join(continentpath,country)
            if country == 'USA':
                df = create_df_usa(countrypath)
            else:
                df = create_df_other(countrypath)
            if df is not None:
                df = df.drop_duplicates().fillna(asnan).reindex(sorted(df.columns), axis=1)
                if country in CountriesMap:
                    country = CountriesMap[country]
                save_df(df, destdir, country, continent)
                
def mapelement(table):
    obss = tf.convert_to_tensor(np.array([[-256.0], [-256.0]]), dtype = tf.float32)
    for key, x in TermMapToMonth.items():
        if key not in table:
            continue
        xy = table[key]
        xy = tf.stack([xy, tf.repeat(x,xy.shape[0])])
        obss = tf.concat([obss,xy], axis = 1) 
    lx = obss[0] > -255.0
    print(obss)
    obss = tf.boolean_mask(obss, lx, axis = 1)
    median = table['Date'].shape[0] // 2
    return {'Date' : table['Date'][median], 'XY' : obss}
