import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def CreateDataFrame(df):
    project_dir = 'C:\\Users\\PLDD\\python\\Python\\ML\\Kaggle\\house_price'
    raw_path = os.path.join(project_dir, 'data', 'raw')
    train_path = os.path.join(raw_path, 'train.csv')
    test_path = os.path.join(raw_path, 'test.csv')
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    test_df['SalePrice'] = 0.0
    df = train_df.append(test_df)
    df.index = list(range(train_df.index.size + test_df.index.size))
    ################# 2916 out of 2919 Utilities are AllPub so i think it should be droped off
    df = df.drop('Utilities', axis=1)
    ############### Handle basment information - replace nan with no_basment or zero quatity
    catbsmflds = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
    ambsmflds = ['BsmtFinSF1', 'BsmtFinSF2', 'TotalBsmtSF', 'BsmtUnfSF', 'BsmtFullBath', 'BsmtHalfBath']
    lx = df[catbsmflds].isnull().all(axis=1)
    lx = lx & (df['TotalBsmtSF'] == 0.0)
    df.loc[lx, catbsmflds] = 'NoBsm'
    df.loc[lx, ambsmflds] = 0.0
    ############### Handle garage information - replace nan with no_garage or zero quatity
    catflds = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']
    yearfld = 'GarageYrBlt'
    amtfld = 'GarageArea'
    lx = (df[catflds].isnull().all(axis=1)) & (df[amtfld] == 0.0)
    df.loc[lx, catflds] = 'NoGarage'
    df.loc[lx, yearfld] = 0
    ############### Handle Pool information - replace nan with no_pool or zero quatity
    catfld = 'PoolQC'
    amtfld = 'PoolArea'
    lx = (df[catfld].isnull()) & (df[amtfld] == 0.0)
    df.loc[lx, catfld] = 'NoPool'
    ############### Handle FireplaceQu nan - it is only no fireplace #################################
    fld = 'FireplaceQu'
    lx = df['Fireplaces'] == 0
    df.loc[lx, fld] = 'NoFireplace'
    ############### Handle 'Fence', 'MiscFeature' - just replace it with NoFence or feature###########
    flds = ['Fence', 'MiscFeature']
    df.loc[:, flds] = df[flds].fillna(value='Noooo')
    return df


##################################################################################################################
def ResolveNaN(df):
    oldflds = []
    newdf = pd.DataFrame()
    ######################## 1) MSZoning - 'RL', 'RM', 'C (all)', 'FV', 'RH', nan ################################
    colname = 'MSZoning'
    oldflds.append(colname)
    #### with Mitchel neighbor it is highly likely that zoning is 'RL' (grouping by Neighbors + BuiltYear)
    lx = (df[colname].isnull()) & (df['Neighborhood'] == 'Mitchel')
    df.loc[lx, colname] = 'RL'
    ### group by IDOTRR neighbor
    invalidlx = df[colname].isnull()
    idotrrlx = df['Neighborhood'] == 'IDOTRR'

    tempdf = pd.get_dummies(df[colname], prefix=colname).astype(float)
    tempdf.loc[invalidlx, :] = tempdf[idotrrlx].sum().values / tempdf[idotrrlx].sum().sum()
    newdf[tempdf.columns] = tempdf
    ######################## 2) LotFrontage - float value, nan ################################
    # Exclude this field from ML - base on the idea that Lot Area is more important parameter than LotFrontage
    colnames = ['LotFrontage', 'GarageCars']
    oldflds += colnames
    ######################## 3) Alley  - nan, 'Grvl', 'Pave' ################################
    colname = 'Alley'
    oldflds.append(colname)
    tempdf = pd.get_dummies(df[colname], prefix=colname, dummy_na=True).astype(float)
    newdf[tempdf.columns] = tempdf
    ######################## 4) 'Exterior1st', 'Exterior2nd','MasVnrType','Electrical','KitchenQual'#############
    ########################'BsmtQual', 'BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2'##################
    ######################## 'Functional','GarageType', 'GarageFinish', 'GarageQual','GarageCond'################
    ######################## 'PoolQC','SaleType'#################################################################
    colnames = ['Exterior1st', 'Exterior2nd', 'MasVnrType', 'Electrical', 'KitchenQual',
                'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
                'Functional', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
                'PoolQC', 'SaleType']
    for colname in colnames:
        lx = df[colname].isnull()
        oldflds.append(colname)
        tempdf = pd.get_dummies(df[colname], prefix=colname).astype(float)
        tempdf.loc[lx, :] = tempdf[~lx].sum().values / tempdf[~lx].sum().sum()
        newdf[tempdf.columns] = tempdf
    ####################### 5) Mean apprx - 'MasVnrArea','BsmtFullBath','BsmtHalfBath'###########################
    colnames = ['MasVnrArea', 'BsmtFullBath', 'BsmtHalfBath']
    for colname in colnames:
        lx = df[colname].isnull()
        df.loc[lx, colname] = df[~lx][colname].mean()

    ####################### 6) Median apprx - 'BsmtFinSF1','BsmtFinSF2','TotalBsmtSF','BsmtUnfSF','GarageArea'###
    colnames = ['BsmtFinSF1', 'BsmtFinSF2', 'TotalBsmtSF', 'BsmtUnfSF', 'GarageArea']
    for colname in colnames:
        lx = df[colname].isnull()
        df.loc[lx, colname] = df[~lx][colname].median()
    ##################### 7) 'GarageYrBlt'#######################################################################
    colname = 'GarageYrBlt'
    lx = df[colname].isnull()
    df.loc[lx, colname] = df[lx]['YearBuilt']

    df = df.drop(labels=oldflds, axis=1)
    df[newdf.columns] = newdf
    ################################################Convert int and Objects type fields into float64 ###############
    df = pd.get_dummies(df).astype('float64')
    df.loc[:, 'Id'] = df['Id'].astype('int32')
    return df


####################################################################################################################
def FormatAndSave(df):
    dff = df.copy()
    categorical_flds = dff.columns[dff.dtypes == 'category'].values
    dff = pd.get_dummies(dff, columns=categorical_flds)

    processed_data_path = os.path.join(os.path.pardir, 'data', 'processed')
    write_train_path = os.path.join(processed_data_path, 'train.csv')
    write_test_path = os.path.join(processed_data_path, 'test.csv')

    # train data
    dff.loc[dff.SalePrice != 0].to_csv(write_train_path)
    # test data
    columns = [column for column in dff.columns if column != 'SalePrice']
    dff.loc[dff.SalePrice == 0, columns].to_csv(write_test_path)
    return df


if __name__ == '__main__':
    df = pd.DataFrame()
    df = df.pipe(CreateDataFrame).pipe(ResolveNaN).pipe(FormatAndSave)
