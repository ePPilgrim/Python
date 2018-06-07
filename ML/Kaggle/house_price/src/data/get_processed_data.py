import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def CreateDataFrame():
    project_dir = 'C:\\Users\\PLDD\\python\\Python\\ML\\Kaggle\\house_price'
    raw_path = os.path.join(project_dir,'data','raw')
    train_path = os.path.join(raw_path, 'train.csv')
    test_path = os.path.join(raw_path, 'test.csv')
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    test_df['SalePrice'] = 0.0
    df = train_df.append(test_df)
    df.index = list(range(train_df.index.size + test_df.index.size))
    df = df.drop('Utilities', axis = 1)
    return df

def ProccessOutliers(df):
    lx = (df['LotFrontage'] > 200) | (df['LotArea'] > 100000) | (df['GrLivArea'] > 4000)
    df.loc[df['LotFrontage'] > 200,'LotFrontage'] = np.nan
    df.loc[df['LotArea'] > 100000,'LotArea'] = np.nan
    df.loc[df['GrLivArea'] > 4000,'GrLivArea'] = np.nan
    return df

def ReplaceNanValues_LotFrontageArea(df, alables, abins, fromfld, tofld, catclass, targetclass):
    fullclass = catclass + [targetclass]
    df[tofld]=pd.cut(x = df[fromfld].values, bins = abins, right = False, labels = alables)
    validdf = df[fullclass].dropna(axis = 0, how = 'any')
    nandf = df[df[targetclass].isnull()]
    pvt = validdf.pivot_table(values = targetclass, index = catclass, aggfunc = np.median)
    pvt=pvt[pvt.notnull().all(1)]
    t1 = pvt.loc[[tuple(x) for x in nandf[catclass].values]]
    t1.index = nandf.index
    df.loc[nandf.index,targetclass] = t1[targetclass]
    df = df.drop(tofld, axis = 1)
    return df

def ProccessNanValues(df):
    DefSeqCat = ['Alley','BsmtQual', 'BsmtCond', 'BsmtExposure','BsmtFinType1', 'BsmtFinType2', 'FireplaceQu',
                 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC',
                 'Fence', 'MiscFeature']
    UndefSeqCat1 = ['MSZoning', 'Utilities', 'Exterior1st', 'Exterior2nd','Electrical', 'KitchenQual',
                    'Functional', 'SaleType'] # there are defenetly lost elements 
    UndefSeqCat2 = ['MasVnrType'] # i may just not make it clear the property of the column
    UndefSeqNum = ['LotFrontage', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 
                   'BsmtFullBath', 'BsmtHalfBath', 'GarageYrBlt', 'GarageCars', 'GarageArea']
    replace=dict.fromkeys(DefSeqCat, 'None')
    df = df.fillna(replace)
#'MasVnrType'
    ix = df.index[df['MasVnrType'].isnull()]
    df.loc[ix,'MasVnrType'] = 'None'
    df.loc[ix,'MasVnrArea'] = 0
    NanCol = ['LotFrontage', 'MSZoning', 'Utilities', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
              'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
              'Electrical', 'BsmtFullBath', 'BsmtHalfBath', 'KitchenQual',
              'Functional', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'SaleType']
    #Replace NAN values in 'LotFrontage' and 'LotArea' with iterpolated value.
    yearlable = ['1906', '1942', '1975', '1991', '2011']
    yearbin = [1800,1906,1942,1975, 1991, 2011]
#case 1
    catclass = ['Neighborhood', 'LotConfig','LotShape','YearInt']
    df = ReplaceNanValues_LotFrontageArea(df, yearlable, yearbin, 'YearBuilt', 'YearInt', catclass, 'LotFrontage' )
    df = ReplaceNanValues_LotFrontageArea(df, yearlable, yearbin, 'YearBuilt', 'YearInt', catclass, 'LotArea')
    #print(df[df['LotFrontage'].isnull()].index.size)
    catclass = ['LotConfig','LotShape','YearInt']
#case 2
    df = ReplaceNanValues_LotFrontageArea(df, yearlable, yearbin, 'YearBuilt', 'YearInt', catclass, 'LotFrontage')
    df = ReplaceNanValues_LotFrontageArea(df, yearlable, yearbin, 'YearBuilt', 'YearInt', catclass, 'LotArea')
    #print(df[df['LotFrontage'].isnull()].index.size)
#case 3
    yearlable = ['1942', '1991', '2011']
    yearbin = [1800, 1942, 1991, 2011]
    df = ReplaceNanValues_LotFrontageArea(df, yearlable, yearbin, 'YearBuilt', 'YearInt', catclass, 'LotFrontage')
    df = ReplaceNanValues_LotFrontageArea(df, yearlable, yearbin, 'YearBuilt', 'YearInt', catclass, 'LotArea')
    #print(df[df['LotFrontage'].isnull()].index.size)
#case 4
    yearlable = ['1975', '1991', '2011']
    yearbin = [1800, 1975, 1991, 2011]
    df = ReplaceNanValues_LotFrontageArea(df, yearlable, yearbin, 'YearBuilt', 'YearInt', catclass, 'LotFrontage')
    df = ReplaceNanValues_LotFrontageArea(df, yearlable, yearbin, 'YearBuilt', 'YearInt', catclass, 'LotArea')
    #print(df[df['LotFrontage'].isnull()].index.size)
#case 5
    yearlable = ['2011']
    yearbin = [1800, 2011]
    df = ReplaceNanValues_LotFrontageArea(df, yearlable, yearbin, 'YearBuilt', 'YearInt', catclass, 'LotFrontage')
    df = ReplaceNanValues_LotFrontageArea(df, yearlable, yearbin, 'YearBuilt', 'YearInt', catclass, 'LotArea')   
#Replace Nan in 'MSZoning'
    ind1 = [1915, 2216, 2250] # C - assign by many factors
    ind2 = [2904] # RL - by Lot Area, LotFrontage and YearBuilt 
    df.loc[ind1,'MSZoning'] = 'C'
    df.loc[ind2,'MSZoning'] = 'RL'    
#Replace Nan in 'Exterior1st' and 'Exterior2nd'
    targetclass = ['Exterior1st', 'Exterior2nd']
    ind = [2151]
    df.loc[ind,targetclass] = 'AsbShng'  
#Add column 'RangeYrBlt' and remove columns 'GarageYrBlt' and 'YearBilt'
    yearbins = [1800, 1895, 1905, 1917, 1927, 1937,1947,1960, 1973, 1989, 1998, 2012]
    yearlabels = ['1895', '1905', '1917', '1927', '1937', '1947', '1960', '1973', '1989', '1998', '2012']
    if df.columns.isin(['YearBuilt']).any():
        df['RangeYrBlt']=pd.cut(x = df['YearBuilt'].values, bins = yearbins, right = False, labels = yearlabels)
        df = df.drop('YearBuilt', axis = 1)
    if df.columns.isin(['GarageYrBlt']).any():
        df = df.drop('GarageYrBlt', axis = 1)
    df.loc[2120,['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF']] = 0
    df.loc[1379, 'Electrical'] = 'SBrkr' #  all houses in 2012 year built have this type of Electrical
    if df.columns.isin(['GrLivArea']).any():
        df = df.drop('GrLivArea', axis = 1) # it is sum of '1stFlrSF' and '2ndFlrSF'
    df.loc[[2120,2188],['BsmtFullBath', 'BsmtHalfBath']] = 0
    df.loc[1555, 'KitchenQual'] = 'TA'
    df.loc[[2216, 2473],'Functional'] = 'Typ'
    df.loc[2576,'GarageArea'] = 0
    if df.columns.isin(['GarageCars']).any():
        df = df.drop('GarageCars', axis = 1) # it has straight dependency on garage area
    df.loc[2489, 'SaleType'] = 'WD' # the most used type according to sold year and sale condition
    return df

def sorted_columns(df, what, bby):
    t_df = df.loc[df[bby] != 0]
    t_df = t_df.groupby([what]).agg({bby : 'median'})
    t_df = t_df.sort_values(by = bby, axis = 0 )
    return t_df.index.values

def FormatAndSave(df):
    dff = df.copy()
    categorical_flds = dff.columns[dff.dtypes == 'category' ].values
    dff = pd.get_dummies(dff)#, columns = categorical_flds)
    processed_data_path = os.path.join(os.path.pardir,'data','processed')
    write_train_path = os.path.join(processed_data_path, 'train.csv')
    write_test_path = os.path.join(processed_data_path, 'test.csv')
    dff.loc[dff.SalePrice != 0].to_csv(write_train_path) 
    columns = [column for column in dff.columns if column != 'SalePrice']
    dff.loc[dff.SalePrice == 0, columns].to_csv(write_test_path) 

if __name__ == '__main__':
    df = CreateDataFrame()
    df = df.pipe(ProccessOutliers).pipe(ProccessNanValues).pipe(FormatAndSave)