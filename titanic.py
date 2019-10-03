import os

import numpy as np
import pandas
import pickle
#import seaborn as sns

from sklearn.pipeline import Pipeline, make_union, make_pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, FunctionTransformer, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

kf = KFold(n_splits=5, shuffle=True, random_state=421)

import matplotlib.pyplot as plt

# simple UI using Argparse

import argparse
import re

def tensorflow_model_valid(s, pat=re.compile(r"^.*\.ckpt$")):
    if not pat.match(s):
        raise argparse.ArgumentTypeError
    return s

parser = argparse.ArgumentParser(description='Using Tensorflow and and SciKit-Learn \
                                              for solving Kaggle`s Titanic problem.')

parser.add_argument('-tf', '--tensor-flow', help='use Tensorflow', action='store_true')
parser.add_argument('-d', '--data', help='path to input data', required=True)
parser.add_argument('-m', '--model', help='path to the model (must end with .ckpt)', \
                                            required=True, type=tensorflow_model_valid)
    

def load_data_for_kaggle(data_path='DATA'):
    
    data_train = pandas.read_csv(os.path.join(data_path, 'train.csv'))
    data_test = pandas.read_csv(os.path.join(data_path, 'test.csv'))

    X_train = data_train.drop('Survived', axis=1)
    y_train = data_train['Survived'].values
    X_test = data_test
    
    return (X_train, y_train, X_test)

(X_train, y_train, X_test) = load_data_for_kaggle()


class FamilyTransform(TransformerMixin):

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        
        X['FamilySize'] = X['Parch'] + X['SibSp'] + 1

        X['Singleton'] = X['FamilySize'].map(lambda s: 1 if s == 1 else 0)
        X['SmallFamily'] = X['FamilySize'].map(lambda s: 1 if 2 <= s <= 4 else 0)
        X['BigFamily'] = X['FamilySize'].map(lambda s: 1 if 5 <= s else 0)
        
        #X.drop(['Parch', 'SibSp'], axis=1, inplace=True)
        
        return X
    

class EmbarkedTransform(TransformerMixin):

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        
        X['Embarked'].fillna('S', inplace=True)

        embarked_dummies = pandas.get_dummies(X['Embarked'], prefix='Embarked')
        X = pandas.concat([X, embarked_dummies], axis=1)
        X.drop('Embarked', axis=1, inplace=True)
        
        return X
    

class CabinTransform(TransformerMixin):
    
    def fit(self, X, y=None):
        
        return self
    
    def transform(self, X, y=None):
        
        X['Cabin'].fillna('U', inplace=True)
        X['Cabin'] = X['Cabin'].map(lambda c: c[0])

        cabin_dummies = pandas.get_dummies(X['Cabin'], prefix='Cabin')
        X = pandas.concat([X, cabin_dummies], axis=1)

        X.drop('Cabin', axis=1, inplace=True)
        
        return X
    

titles_dictionary = {
    'Capt': 'Officer',
    'Col': 'Officer',
    'Major': 'Officer',
    'Jonkheer': 'Royalty',
    'Don': 'Royalty',
    'Sir': 'Royalty',
    'Dr': 'Officer',
    'Rev': 'Officer',
    'the Countess': 'Royalty',
    'Mme': 'Mr',
    'Mlle': 'Miss',
    'Ms': 'Mrs',
    'Mr': 'Mr',
    'Mrs': 'Mrs',
    'Miss': 'Miss',
    'Master': 'Master',
    'Lady': 'Royalty',
}

class TitleTransform(TransformerMixin):

    def fit(self, X, y=None):
        
        self.titles = ['Title_{}'.format(s) for s in list(set(titles_dictionary.values()))]
        
        return self
    
    def transform(self, X, y=None):
        
        X['Title'] = X['Name'].map(lambda name: name.split(',')[1].split('.')[0].strip())
        X['Title'] = X.Title.map(titles_dictionary)
        

        return X


class AgeTransform(TransformerMixin):

    def fit(self, X, y=None):
        grouped_data = X.groupby(['Sex', 'Pclass', 'Title'])
        self.grouped_median_age = grouped_data.median()
        self.grouped_median_age = self.grouped_median_age.reset_index()[['Sex', 'Pclass', 'Title', 'Age']]
        
        return self
        
    def transform(self, X, y=None):
        
        X['Age'] = X.apply(lambda row: self._fill_age_(row) if np.isnan(row['Age']) else row['Age'], axis=1)

        return X
    
    def _fill_age_(self, row):
        condition = (
            (self.grouped_median_age['Sex'] == row['Sex']) &
            (self.grouped_median_age['Pclass'] == row['Pclass']) &
            (self.grouped_median_age['Title'] == row['Title'])
        )
        if np.isnan(self.grouped_median_age[condition]['Age'].values[0]):
            condition = (
                (self.grouped_median_age['Sex'] == row['Sex']) &
                (self.grouped_median_age['Pclass'] == row['Pclass'])
            )

        return self.grouped_median_age[condition]['Age'].values[0]
    
    
class NameTransform(TransformerMixin):

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        
        X.drop('Name', axis=1, inplace=True)
    
        titles_dummies = pandas.get_dummies(X['Title'], prefix='Title')
        X = pandas.concat([X, titles_dummies], axis=1)
        X.drop('Title', axis=1, inplace=True)

        return X
    
    
class BinaryTransform(TransformerMixin):
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        
        X['Sex'] = X['Sex'].map(lambda sex: 1 if sex == 'male' else 0)
        
        return X
    
    
class DroppingTranform(TransformerMixin):
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        
        # Parameters that have the biggest weights, while others can be ignored
        important_cols = ['Title_Mr', 'Sex', 'Ticket', 'Fare', 'PassengerId', \
                          'Age', 'Pclass', 'FamilySize', 'Cabin_T', 'BigFamily', \
                          'Title_Miss', 'SibSp', 'Title_Officer']
    
        df = pandas.DataFrame()
        for col in important_cols:
            df[col] = X[col]
            
        return df
    
    
preprocess_pipeline = Pipeline(steps=[
    ('family', FamilyTransform()),
    ('embarked', EmbarkedTransform()),
    ('cabin', CabinTransform()),
    ('title', TitleTransform()),
    ('age', AgeTransform()),
    ('name', NameTransform()),
    ('binary', BinaryTransform()),
    ('drop', DroppingTransform()),
])
    

if __name__ == "__main__":
    
    args = vars(parser.parse_args())
    X_test = pd.read_csv(args["data"])
    X_test = preprocess_pipeline.transform(X_test)
    
    if args['tensor_flow'] == True:
        saver = tf.train.import_meta_graph(args["model"] + ".meta")
        with tf.Session() as sess:
            saver.restore(sess, args["model"])
            y_pred = sess.run(tf.cast(tf.round(outputs), dtype=tf.int32), feed_dict={X: X_test})
        
    else:
        model = pickle.load(open(args['model'], 'rb'))
        y_pred = model.predict(X_test)
        y_pred.to_csv("output.csv", index=False)

    output = pandas.DataFrame(data=y_pred, columns=["Survived"])
    output.to_csv('output.csv', index=False)
