import os
import pldd_transformation as pt
import pandas as pd
import re
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing.data import QuantileTransformer
from sklearn.linear_model import ElasticNet, Lasso, LassoLars,  BayesianRidge, LassoLarsIC, Ridge


def get_raw_data(raw_path):
    train_path = os.path.join(raw_path, 'train.csv')
    test_path = os.path.join(raw_path, 'test.csv')
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    test['SalePrice'] = 0.0
    train['Id'] = 0
    test['Id'] = 1
    return train.append(test)


def get_processed_data(df):
    tr1 = pt.make_pipe_decorator(pt.Transformer1('GarageArea', 0, {0: ['GarageYrBlt', 'GarageCars'],
                                                                   'None': ['GarageType', 'GarageFinish', 'GarageQual',
                                                                            'GarageCond']}),
                                 pt.Transformer2(['GarageCars'], ['GarageArea']),
                                 pt.Transformer3(['YearRemodAdd'], ['GarageYrBlt']),
                                 pt.Transformer4(['BuiltAge', 'RenowateAge', 'GarageAge'],
                                                 ['YrSold'] * 3, ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt'],
                                                 ['MoSold']),
                                 pt.Transformer5(['GarageFinish', 'GarageQual', 'GarageCond']),
                                 pt.Transformer2(aMeanFlds=['GarageFinish', 'GarageQual', 'GarageCond']),
                                 pt.Transformer7(['GarageType']))

    tr2 = pt.make_pipe_decorator(
        pt.Transformer1('TotalBsmtSF', 0, {0: ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
                                               'BsmtFullBath', 'BsmtHalfBath'],
                                           'None': ['BsmtQual', 'BsmtCond', 'BsmtExposure',
                                                    'BsmtFinType1', 'BsmtFinType2']}),
        pt.Transformer1('MasVnrArea', 0, {'None': ['MasVnrType']}),
        pt.Transformer1('MasVnrType', 'None', {0: ['MasVnrArea']}),
        pt.Transformer8(['MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2'], ['MasVnrType', 'BsmtFinType1', 'BsmtFinType2']),
        pt.Transformer9('TotalBsmtSF', ['BsmtFinSF1', 'BsmtFinSF2'], [1.0, 1.0]),
        pt.Transformer9('BsmtUnfSF', ['TotalBsmtSF', 'BsmtFinSF1', 'BsmtFinSF2'], [1.0, -1.0, -1.0]),
        pt.Transformer2(aMeanFlds=['BsmtFullBath', 'BsmtHalfBath']),
        pt.Transformer5(['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']),
        pt.Transformer2(aMeanFlds=['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']),
        pt.Transformer7(['MasVnrType'])
        )

    tr3 = pt.make_pipe_decorator(pt.FillNa(['FireplaceQu', 'Alley', 'PoolQC', 'Fence', 'MiscFeature']),
                                 pt.DropFields(['Utilities']),
                                 pt.Transformer5(['FireplaceQu', 'Alley', 'PoolQC']), pt.TurnObjIntoNum())

    # 4. ['MSZoning','Neighborhood','LotFrontage','LotConfig','LotArea']
    param_grid = {'alpha': [0.2, 0.1, 1e-2, 1e-3, 1e-4]}
    solver = GridSearchCV(Lasso(max_iter=5000, tol=0.0001), cv=5, param_grid=param_grid)
    tr4 = pt.make_pipe_decorator(
        pt.Transformer6(solver, 'LotFrontage', ['Neighborhood', 'LotFrontage', 'LotConfig', 'LotArea'],
                        RobustScaler(),
                        QuantileTransformer()),
        pt.Transformer10(), pt.TurnObjIntoNum())

    # 5. ['ExterQual','ExterCond','HeatingQC','LandSlope','LotShape','PavedDrive','Street','CentralAir','KitchenQual']
    tr5 = pt.make_pipe_decorator(pt.Transformer5(['ExterQual', 'ExterCond', 'HeatingQC', 'LandSlope', 'LotShape',
                                                  'PavedDrive', 'Street', 'CentralAir', 'KitchenQual']),
                                 pt.Transformer2(aMeanFlds=['KitchenQual']))

    # 6. ['Exterior1st','Exterior2nd','Electrical','SaleType','Functional']
    tr6 = pt.make_pipe_decorator(
        pt.Transformer7(['Exterior1st', 'Exterior2nd', 'Electrical', 'SaleType', 'Functional']))

    # the rest fields
    tr7 = pt.make_pipe_decorator(pt.TurnObjIntoNum())

    trs = ColumnTransformer(
        [("tr1", tr1, ['GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual',
                       'GarageCond', 'YearBuilt', 'YearRemodAdd', 'MoSold', 'YrSold']),
         ("tr2", tr2, ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2',
                       'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea',
                       'MasVnrType']),
         ("tr3", tr3, ['FireplaceQu', 'Alley', 'PoolQC', 'Fence', 'MiscFeature', 'Utilities']),
         ("tr4", tr4, ['MSZoning', 'Neighborhood', 'LotFrontage', 'LotConfig', 'LotArea']),
         ("tr5", tr5, ['ExterQual', 'ExterCond', 'HeatingQC', 'LandSlope', 'LotShape',
                       'PavedDrive', 'Street', 'CentralAir', 'KitchenQual']),
         ("tr6", tr6, ['Exterior1st', 'Exterior2nd', 'Electrical', 'SaleType', 'Functional'])
         ], remainder=tr7)
    tr = make_pipeline(pt.TreatOutliers(), trs)
    X = tr.fit_transform(df)
    column_names = [re.compile(r'^\w*__').sub('', s) for s in trs.get_feature_names()]
    return pd.DataFrame(data=X, columns=column_names).astype('float')


def FormatAndSave(raw_path, processed_path):
    df = get_raw_data(raw_path)
    df = get_processed_data(df)
    write_train_path = os.path.join(processed_path, 'train.csv')
    write_test_path = os.path.join(processed_path, 'test.csv')

    # train data
    df.loc[df.Id < 1].to_csv(write_train_path)
    # test data
    columns = [column for column in df.columns if column != 'SalePrice']
    df.loc[df.Id > 0, columns].to_csv(write_test_path)
    return df


if __name__ == '__main__':
    raw_path = os.path.join(os.path.pardir, os.path.pardir, 'data', 'raw')
    processed_path = os.path.join(os.path.pardir, os.path.pardir, 'data', 'processed')
    FormatAndSave(raw_path, processed_path)
    print("Get_processed_data")
