import xgboost as xgb
from sklearn.dummy import DummyClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression


class classification_models:
    def __init__(self):
        self.baseline = self.baseline()
        self.svc = self.svc()
        self.naive_bayes = self.naive_bayes()
        self.knn = self.knn()
        self.logistic_regression = self.logistic_regression()
        self.xgboost = self.xgboost()


class baseline:
    model = DummyClassifier()

    parameters = {
                   'model__strategy': ['uniform'],
                   'model__random_state': [8]
                 }

    @staticmethod
    def best_model(params):
        clf = DummyClassifier(**params)
        return clf


class svc:
    model = svm.SVC()

    parameters = {'model__C': [0.1, 1, 10, 100],
                  'model__gamma': [1, 0.1, 0.01, 0.001],
                  'model__kernel': ['rbf', 'poly', 'sigmoid']}

    @staticmethod
    def best_model(params):
        clf = svm.SVC(**params)
        return clf


class naive_bayes:
    model = GaussianNB()

    parameters = {
                'model__var_smoothing': [1e-2, 1e-4, 1e-6,1e-8,1e-10,1e-12,1e-14,1e-16]
                 }

    @staticmethod
    def best_model(params):
        clf = GaussianNB(**params)
        return clf

class knn:
    model = KNeighborsClassifier()

    parameters = {'model__n_neighbors': [3, 4, 5, 10, 15, 20],
                  'model__weights': ['uniform', 'distance'],
                  'model__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                  'model__leaf_size': [5, 10, 20, 30, 50]
                  }

    @staticmethod
    def best_model(params):
        clf = KNeighborsClassifier()
        return clf

class logistic_regression:
    model = LogisticRegression()

    parameters = {
                   'model__penalty': ['l1', 'l2', 'elasticnet', 'none'],
                   'model__C': [1e-2, 1e-4, 0.01, 0.1,0.3, 0.5,0.8, 1],
                   'model__solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                   'model__random_state': [8]
                 }

    @staticmethod
    def best_model(params):
        clf = LogisticRegression(**params)
        return clf

class xgboost:
    model = xgb.XGBClassifier()

    parameters = {
                   'model__learning_rate': [0.01,0.1,0.5,1],
                   'model__max_depth': [2,4,8,12,16,20,30],
                   'model__min_samples_leaf': [2,5,10,30,60,90,120],
                   'model__min_samples_split': [2,5,10,30,60,90,120],
                   'model__subsample': [0.5,0.8,1],
                   'model__n_estimators': [5,10,15,25,50],
                   'model__random_state': [16]
                 }

    @staticmethod
    def best_model(params):
        clf = xgb.XGBClassifier(**params)
        return clf

