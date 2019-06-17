import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV


class BaseTransformer:
    def fit(self, x, y=None):
        return self

    def fit_transform(self, X, y=None):
        self = self.fit(X)
        return self.transform(X)

class Transformer1(BaseTransformer):
    def __init__(self, aKeyfld, aKeyVal, aSets):
        self.sets = aSets
        self.keyfld = aKeyfld
        self.keyval = aKeyVal

    def transform(self, X):
        lx = (X[self.keyfld] == self.keyval).values.reshape(-1, 1)
        for key in self.sets:
            masklx = lx & np.array([[el in self.sets[key] for el in X.columns]])
            X = X.mask(masklx, other=key)
        return X


class Transformer2(
    BaseTransformer):  # 'BsmtFullBath','BsmtHalfBath' and cat fields try to generilize this function with Transformation7
    def __init__(self, aMeanFlds=[], aMedianFlds=[]):
        self.meanflds = aMeanFlds
        self.medianflds = aMedianFlds

    def transform(self, X):
        concv = X[self.meanflds].mean(axis=0).append(X[self.medianflds].median(axis=0))
        concf = self.meanflds + self.medianflds
        lx = X[concf].isnull().values
        X[concf] = X[concf].mask(lx, np.broadcast_to(concv, lx.shape))
        return X


class Transformer3(BaseTransformer):
    def __init__(self, aFromFlds, aToFlds):
        self.fromflds = aFromFlds
        self.toflds = aToFlds

    def transform(self, X):
        lx = X[self.toflds].isnull().values
        if lx.any():
            X[self.toflds] = X[self.toflds].mask(lx, X[self.fromflds].values)
        return X


class Transformer4(BaseTransformer):
    def __init__(self, aResFlds, aXFlds, aYFlds, aDropFlds):
        self.zflds = aResFlds
        self.xflds = aXFlds
        self.yflds = aYFlds
        self.dropflds = aDropFlds

    def transform(self, X):
        X[self.yflds] = X[self.xflds] - X[self.yflds].values
        X.columns = [self.zflds[self.yflds.index(el)] if el in self.yflds else el for el in X.columns]
        return X.drop(columns=self.dropflds)


class Transformer5(BaseTransformer):
    def __init__(self, aFlds):
        self.sortmap = {'None': ['Po', 'Grvl', 'No', 'Unf'], 'Po': 'Fa', 'Fa': 'TA', 'TA': 'Gd', 'Gd': 'Ex', 'Ex': [],
                        'Sev': 'Mod', 'Mod': 'Gtl', 'Gtl': [],
                        'Reg': 'IR1', 'IR1': 'IR2', 'IR2': 'IR3', 'IR3': [],
                        'N': 'P', 'P': 'Y', 'Y': [],
                        'Grvl': 'Pave', 'No': 'Mn',
                        'Pave': [], 'Mn': 'Av', 'Av': 'Gd',
                        'Unf': ['LwQ', 'RFn'], 'LwQ': 'Rec', 'Rec': 'BLQ', 'BLQ': 'ALQ', 'ALQ': 'GLQ', 'GLQ': [],
                        'RFn': 'Fin', 'Fin': []}
        self.flds = aFlds

    def __sort_fields(self, values):
        sortedseq = []
        values = list(values)
        for i in range(len(values)):
            if values[i] is None:
                continue
            subseq = [values[i]]
            subsortseq = []
            for key in subseq:
                if key in values:
                    values[values.index(key)] = None
                    subsortseq.append(key)
                    subseq += self.sortmap[key] if type(self.sortmap[key]) is list else [self.sortmap[key]]
            sortedseq = subsortseq + sortedseq
        return sortedseq

    def transform(self, X):
        for fld in self.flds:
            uniquevals = X[~X[fld].isnull().values][fld].unique()
            X[fld] = pd.Categorical(X[fld].values, self.__sort_fields(uniquevals), ordered=True).codes
        lxm = X[self.flds] < 0.0
        X[self.flds] = X[self.flds].mask(lxm, np.nan)
        return X


# find 'LotFrontage' from 'LotFrontage','Neighborhood','LotConfig','LotArea'
class Transformer6(BaseTransformer):
    def __init__(self, aLinearModel, aYFld, aXFlds, aXScaler, aYScaler, aParamGrid=None, aCV=5):
        if aParamGrid is None:
            self.solver = aLinearModel
        else:
            self.solver = GridSearchCV(estimator=aLinearModel, cv=aCV, param_grid=aParamGrid)
        self.xscaler = aXScaler
        self.yscaler = aYScaler
        self.yfld = aYFld
        self.xflds = aXFlds

    def transform(self, X):
        df = pd.get_dummies(X[self.xflds])
        lx = df[self.yfld].isnull().values
        if lx.any():
            y = df[~lx][self.yfld].values.reshape(-1, 1)
            self.yscaler = self.yscaler.fit(y)
            x = df.drop(self.yfld, axis=1).values.astype('float')
            x_train = x[~lx]
            x_test = x[lx]
            self.xscaler = self.xscaler.fit(x_train)
            x_train = self.xscaler.transform(x_train)
            x_test = self.xscaler.transform(x_test)
            y = self.yscaler.transform(y)
            self.solver.fit(x_train, y)
            y = self.solver.predict(x_test).astype('float').reshape(-1, 1)
            X.loc[lx, self.yfld] = self.yscaler.inverse_transform(y)
            # x_train, x_test, y_train, y_test = train_test_split(x[~lx], y, test_size=0.4, random_state=0)
            # self.scaler = self.scaler.fit(x_train)
            # x_train = self.scaler.transform(x_train)
            # x_test = self.scaler.transform(x_test)
            # self.solver.fit(x_train, y_train)
        return X


# e.g. 'Exterior1st','Exterior2nd','MasVnrType','Electrical','GarageType','SaleType','Functional'
class Transformer7(BaseTransformer):
    def __init__(self, aCatFlds):
        self.flds = aCatFlds

    def transform(self, X):
        oldflds = []
        for fld in self.flds:
            lx = X[fld].isnull()
            oldflds.append(fld)
            tmp = pd.get_dummies(X[[fld]], prefix=fld).astype(float)
            tmp.loc[lx, :] = tmp.mean(axis=0).values
            X = pd.concat([X, tmp], axis=1)
        X = X.drop(labels=oldflds, axis=1)
        return X


# ['MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2'] from ['MasVnrType', 'BsmtFinType1','BsmtFinType2']
class Transformer8(BaseTransformer):
    def __init__(self, aNumFlds, aCatFlds):
        self.numflds = aNumFlds
        self.catflds = aCatFlds

    def transform(self, X):
        for (catfld, numfld) in zip(self.catflds, self.numflds):
            lx = X[numfld].isnull()
            tmp = X.groupby([catfld])[numfld].describe()
            X.loc[lx, numfld] = np.sum(tmp['50%'] * tmp['count'] / tmp['count'].sum())
        return X


class Transformer9(BaseTransformer):
    def __init__(self, aResFld, aXFlds, aXSigns):
        self.zfld = aResFld
        self.xflds = aXFlds
        self.xsigns = np.array(aXSigns)

    def transform(self, X):
        lx = X[self.zfld].isnull()
        X.loc[lx, self.zfld] = (X.loc[lx, self.xflds] * self.xsigns).sum(axis=1)
        return X


class Transformer10(BaseTransformer):  # only for fields 'MSZoning','Neighborhood'
    def transform(self, X):
        lx = (X['MSZoning'].isnull()) & (X['Neighborhood'] == 'Mitchel')
        X.loc[lx, 'MSZoning'] = 'RL'
        invalidlx = X['MSZoning'].isnull()
        idotrrlx = X['Neighborhood'] == 'IDOTRR'
        tempdf = pd.get_dummies(X['MSZoning'], prefix='MSZoning').astype(float)
        tempdf.loc[invalidlx, :] = tempdf[idotrrlx].sum().values / tempdf[idotrrlx].sum().sum()
        X = pd.concat([X, tempdf], axis=1)
        return X.drop(labels='MSZoning', axis=1)


class TreatOutliers(BaseTransformer):  # only for outliers
    def transform(self, X):
        lx = (X['SalePrice'] <= 200000) & (X['GrLivArea'] >= 4000)
        lx |= X['LotFrontage'] > 300
        lx &= X['Id'] == 0  # zero mean training set
        return X[~lx]


class FillNa(BaseTransformer):  # at least for 'FireplaceQu','Alley','PoolQC','Fence','MiscFeature'
    def __init__(self, aFlds, aVal='None'):
        self.flds = aFlds
        self.val = aVal

    def transform(self, X):
        X[self.flds] = X[self.flds].fillna(value=self.val)
        return X


class DropFields(BaseTransformer):  # at least for 'Utilities'
    def __init__(self, aFlds):
        self.flds = aFlds

    def transform(self, X):
        return X.drop(self.flds, axis=1)


class TurnObjIntoNum(BaseTransformer):
    def transform(self, X):
        lx = (X.dtypes == 'object') & (~X.isnull().any(axis=0))
        oldflds = X.columns[lx]
        tmp = pd.get_dummies(X[oldflds]).astype(float)
        X = X.drop(labels=oldflds, axis=1)
        return pd.concat([X, tmp], axis=1)


class PipeDecorator(Pipeline):
    def __init__(self, steps, memory=None):
        super().__init__(steps, memory)
        self.flds = None

    def transform(self, x):
        x = super().transform(x)
        self.flds = x.columns
        return x

    def fit(self, X, y=None):
        self.flds = X.columns
        return self

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def get_feature_names(self):
        return self.flds


def make_pipe_decorator(*aSteps):
    pipe = make_pipeline(*list(aSteps))
    return PipeDecorator(pipe.steps)
