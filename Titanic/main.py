# 数据处理
import numpy as np
import pandas as pd
# 绘图
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline
# 各种模型、数据处理方法
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import precision_score
import warnings

train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')
combine_df = pd.concat([train_df, test_df])

# NameLength
train_df.groupby(train_df.Name.apply(lambda x: len(x)))['Survived'].mean().plot()
combine_df['Name_Len'] = combine_df['Name'].apply(lambda x: len(x))
combine_df['Name_Len'] = pd.qcut(combine_df['Name_Len'], 5)
combine_df.groupby(combine_df['Name'].apply(lambda x: x.split(', ')[1]).apply(lambda x: x.split('.')[0]))[
    'Survived'].mean().plot()

# Title
combine_df['Title'] = combine_df['Name'].apply(lambda x: x.split(', ')[1]).apply(lambda x: x.split('.')[0])
combine_df['Title'] = combine_df['Title'].replace(
    ['Don', 'Dona', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col', 'Sir', 'Dr'], 'Mr')
combine_df['Title'] = combine_df['Title'].replace(['Mlle', 'Ms'], 'Miss')
combine_df['Title'] = combine_df['Title'].replace(['the Countess', 'Mme', 'Lady', 'Dr'], 'Mrs')
df = pd.get_dummies(combine_df['Title'], prefix='Title')
combine_df = pd.concat([combine_df, df], axis=1)

combine_df['Fname'] = combine_df['Name'].apply(lambda x: x.split(',')[0])
combine_df['Familysize'] = combine_df['SibSp'] + combine_df['Parch']

# 有女性死亡的家庭
dead_female_Fname = list(set(combine_df[(combine_df.Sex == 'female') & (combine_df.Age >= 12)
                                        & (combine_df.Survived == 0) & (combine_df.Familysize >= 1)]['Fname'].values))
# 有男性存活的家庭
survive_male_Fname = list(set(combine_df[(combine_df.Sex == 'male') & (combine_df.Age >= 12)
                                         & (combine_df.Survived == 1) & (combine_df.Familysize >= 1)]['Fname'].values))
combine_df['Dead_female_family'] = np.where(combine_df['Fname'].isin(dead_female_Fname), 0, 1)
combine_df['Survive_male_family'] = np.where(combine_df['Fname'].isin(survive_male_Fname), 0, 1)

# Name->Title
combine_df = combine_df.drop(['Name', 'Fname'], axis=1)

# 添加一个小孩子标签
group = combine_df.groupby(['Title', 'Pclass'])['Age']
combine_df['Age'] = group.transform(lambda x: x.fillna(x.median()))
combine_df = combine_df.drop('Title', axis=1)
combine_df['IsChild'] = np.where(combine_df['Age'] <= 12, 1, 0)
combine_df['Age'] = pd.cut(combine_df['Age'], 5)
combine_df = combine_df.drop('Age', axis=1)

# 将上面提取过的Familysize再离散化
combine_df['Familysize'] = np.where(combine_df['Familysize'] == 0, 'ALone',
                                    np.where(combine_df['Familysize'] <= 3, 'Normal', 'Big'))
df = pd.get_dummies(combine_df['Familysize'], prefix='Familysize')
combine_df = pd.concat([combine_df, df], axis=1).drop(['SibSp', 'Parch', 'Familysize'], axis=1)

# ticket
combine_df['Ticket_Lett'] = combine_df['Ticket'].apply(lambda x: str(x)[0])
combine_df['Ticket_Lett'] = combine_df['Ticket_Lett'].apply(lambda x: str(x))
combine_df['High_Survival_Ticket'] = np.where(combine_df['Ticket_Lett'].isin(['1', '2', 'P']), 1, 0)
combine_df['Low_Survival_Ticket'] = np.where(combine_df['Ticket_Lett'].isin(['A', 'W', '3', '7']), 1, 0)
combine_df = combine_df.drop(['Ticket', 'Ticket_Lett'], axis=1)

# 缺省的Embarked用S填充
combine_df.Embarked = combine_df.Embarked.fillna('S')
df = pd.get_dummies(combine_df['Embarked'], prefix='Embarked')
combine_df = pd.concat([combine_df, df], axis=1).drop('Embarked', axis=1)

# Cabin
combine_df['Cabin_isNull'] = np.where(combine_df['Cabin'].isnull(), 0, 1)
combine_df = combine_df.drop('Cabin', axis=1)

# Pclass
df = pd.get_dummies(combine_df['Pclass'], prefix='Pclass')
combine_df = pd.concat([combine_df, df], axis=1).drop('Pclass', axis=1)

# Sex
df = pd.get_dummies(combine_df['Sex'], prefix='Sex')
combine_df = pd.concat([combine_df, df], axis=1).drop('Sex', axis=1)

# Fare
combine_df['Fare'].fillna(combine_df['Fare'].dropna().median(), inplace=True)
combine_df['Low_Fare'] = np.where(combine_df['Fare'] <= 8.662, 1, 0)
# combine_df['Mid_Fare'] = np.where((combine_df['Fare']>8.662)&(combine_df['Fare']<=26), 1, 0)
combine_df['High_Fare'] = np.where(combine_df['Fare'] >= 26, 1, 0)
combine_df = combine_df.drop('Fare', axis=1)

# 所有特征转化成数值型编码
features = combine_df.drop(["PassengerId", "Survived"], axis=1).columns
le = LabelEncoder()
for feature in features:
    le = le.fit(combine_df[feature])
    combine_df[feature] = le.transform(combine_df[feature])

if __name__ == '__main__':
    X_all = combine_df.iloc[:891, :].drop(["PassengerId", "Survived"], axis=1)
    Y_all = combine_df.iloc[:891, :]["Survived"]
    X_test = combine_df.iloc[891:, :].drop(["PassengerId", "Survived"], axis=1)


    # logreg = LogisticRegression()
    # svc = SVC()
    # knn = KNeighborsClassifier(n_neighbors=3)
    # decision_tree = DecisionTreeClassifier()
    # random_forest = RandomForestClassifier(n_estimators=300, min_samples_leaf=4, class_weight={0: 0.745, 1: 0.255})
    # gbdt = GradientBoostingClassifier(n_estimators=500, learning_rate=0.03, max_depth=3)
    # xgb = XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.03)
    # clfs = [logreg, svc, knn, decision_tree, random_forest, gbdt, xgb]
    #
    # kfold = 10
    # cv_results = []
    # for classifier in clfs:
    #     cv_results.append(
    #         cross_val_score(classifier, X_all.values, y=Y_all.values, scoring="accuracy", cv=kfold, n_jobs=4))
    #
    # cv_means = []
    # cv_std = []
    # for cv_result in cv_results:
    #     cv_means.append(cv_result.mean())
    #     cv_std.append(cv_result.std())
    #
    # ag = ["LR", "SVC", 'KNN', 'decision_tree', "random_forest", "GBDT", "xgbGBDT"]
    # cv_res = pd.DataFrame({"CrossValMeans": cv_means, "CrossValerrors": cv_std,
    #                        "Algorithm": ag})
    #
    # g = sns.barplot("CrossValMeans", "Algorithm", data=cv_res, palette="Set3", orient="h", **{'xerr': cv_std})
    # g.set_xlabel("Mean Accuracy")
    # g = g.set_title("Cross validation scores")
    # for i in range(7):
    #     print(ag[i], cv_means[i])

    class Bagging(object):

        def __init__(self, estimators):
            self.estimator_names = []
            self.estimators = []
            for i in estimators:
                self.estimator_names.append(i[0])
                self.estimators.append(i[1])
            self.clf = LogisticRegression()

        def fit(self, train_x, train_y):
            for i in self.estimators:
                i.fit(train_x, train_y)
            x = np.array([i.predict(train_x) for i in self.estimators]).T
            y = train_y
            self.clf.fit(x, y)

        def predict(self, x):
            x = np.array([i.predict(x) for i in self.estimators]).T
            # print(x)
            return self.clf.predict(x)

        def score(self, x, y):
            s = precision_score(y, self.predict(x))
            # print(s)
            return s


    lr = LogisticRegression()
    # estimators_range = [200, 250, 300, 350, 400]
    # leaf_range = [2, 3, 4, 5, 6]
    # param_grid = {'n_estimators': estimators_range, 'min_samples_leaf': leaf_range,
    #               'class_weight': [{0: 0.745, 1: 0.255}]}
    rf = RandomForestClassifier(n_estimators=400, min_samples_leaf=2, class_weight={0: 0.745, 1: 0.255})
    # gs = GridSearchCV(estimator=rf, param_grid=param_grid, n_jobs=4, cv=10)
    # gs = gs.fit(X_all, Y_all)
    # print(gs.best_score_)
    # print(gs.best_params_)
    # rf = gs.best_estimator_

    # estimators_range = [400, 450, 500, 550, 600]
    # rate_range = [0.03, 0.035, 0.04, 0.045, 0.05]
    # depth_range = [2, 3, 4, 5]
    # param_grid = {'n_estimators': estimators_range, 'learning_rate': rate_range, 'max_depth': depth_range}
    gbdt = GradientBoostingClassifier(n_estimators=450, learning_rate=0.04, max_depth=3)
    # gs = GridSearchCV(estimator=gbdt, param_grid=param_grid, n_jobs=4, cv=10)
    # gs = gs.fit(X_all, Y_all)
    # print(gs.best_score_)
    # print(gs.best_params_)
    # gbdt = gs.best_estimator_

    # estimators_range = [400, 450, 500, 550, 600]
    # rate_range = [0.03, 0.035, 0.04, 0.045, 0.05]
    # depth_range = [2, 3, 4, 5]
    # param_grid = {'n_estimators': estimators_range, 'learning_rate': rate_range, 'max_depth': depth_range}
    xgbGBDT = XGBClassifier(n_estimators=500, learning_rate=0.04, max_depth=4)
    # gs = GridSearchCV(estimator=xgbGBDT, param_grid=param_grid, n_jobs=4, cv=10)
    # gs = gs.fit(X_all, Y_all)
    # print(gs.best_score_)
    # print(gs.best_params_)
    # xgbGBDT = gs.best_estimator_

    bag = Bagging([('xgb', xgbGBDT), ('lr', lr), ('gbdt', gbdt), ('rf', rf)])
    score = 0
    for i in range(0, 10):
        num_test = 0.20
        X_train, X_cv, Y_train, Y_cv = train_test_split(X_all.values, Y_all.values, test_size=num_test)
        bag.fit(X_train, Y_train)
        # Y_test = bag.predict(X_test)
        acc_xgb = round(bag.score(X_cv, Y_cv) * 100, 2)
        score += acc_xgb
    print(score / 10)

    # Predict
    bag.fit(X_all.values, Y_all.values)
    Y_test = bag.predict(X_test.values).astype(int)
    submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_test
    })
    submission.to_csv(r'submission_eng.csv', index=False)
