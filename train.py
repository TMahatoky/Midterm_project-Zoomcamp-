#!/usr/bin/env python
# coding: utf-8

import pandas as pd 
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from category_encoders import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import pickle 
import warnings
from pandas.core.common import SettingWithCopyWarning

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

# importing and wrangling
df = pd.read_csv("fake_job_postings.csv")

df.shape
df.head()

# choosing the features to be considered for the problem
sample = df[["title", "location", "salary_range", "telecommuting", "has_company_logo", "has_questions", "employment_type", "required_experience", "required_education", "industry", "function", "fraudulent"]]
sample.head()

sample["location"] = sample.location.str.split(",", expand = True)[0]

# dropping feature with high number of NaN
f"{sample.salary_range.isna().sum()} missing out of {sample.shape[0]}\n"
sample.drop(columns = "salary_range", inplace = True)

# Analyzing the target data 
sample.fraudulent.value_counts(normalize = True).plot(kind = "bar")
print("The target is highly imbalanced in favor of 'not fraudulent' ")

# Exploratory data analysis
sample.describe()

sample.corr()

sample.groupby("required_education").count().fraudulent.plot(kind = "bar")
sample.groupby("required_experience").count().fraudulent.plot(kind = "bar")
sample.groupby("location").count().fraudulent.sort_values().tail(10).plot(kind = "bar")

# dropping features with 25% missing values
sample.isna().sum() > (len(sample)/4)
sample.drop(columns = ["required_experience", "required_education", "industry", "function"], inplace = True)
sample.head()

# dropping categorical feature with high cardinality 
sample.title.nunique()
sample.drop(columns = "title", inplace = True)

# dropping missing values as the remaining features have less than 25% that are missing
sample.dropna(inplace = True)
sample.info()

# model building
target = "fraudulent"
y = sample[target]
X = sample.drop(columns = target)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

print("shape of training and test features")
print(X_train.shape, X_test.shape)

# hyperparameter tuning and model training
params= {"svc__kernel":("linear", "poly"), "svc__C":[0.1, 1, 10], "svc__degree":[2, 3]}

clf2 = make_pipeline(
    OneHotEncoder(use_cat_names = True),
    SVC(random_state = 42)
)

model_svc = GridSearchCV(
    clf2,
    param_grid = params,
    cv = 5,
    n_jobs = -1
)

model_svc.fit(X_train, y_train)

params2 = {"randomforestclassifier__n_estimators":[10, 25, 50, 75, 100], "randomforestclassifier__max_depth":[5, 10, 15]}

clf = make_pipeline( 
    OneHotEncoder(use_cat_names = True),
    RandomForestClassifier(random_state = 42)
)

model_forest = GridSearchCV(
    clf, 
    param_grid = params2,
    cv = 5, 
    n_jobs = -1
)

model_forest.fit(X_train, y_train)


model_lreg = make_pipeline(
    OneHotEncoder(use_cat_names = True), 
    LogisticRegression(max_iter = 1000, random_state = 42)
)


model_lreg.fit(X_train, y_train)

# baseline model accuracy score
acc_baseline = y_train.value_counts(normalize = True).max()
print(f"baseline accuracy : {acc_baseline}")


models = [model_svc, model_forest, model_lreg]

accs_train = []
accs_test = []


for val in models: 
    y_pred = val.predict(X_test)
    accs_train.append(val.score(X_train, y_train))
    accs_test.append(val.score(X_test, y_test))
    print(f"training {val} \n")

print(f"training accuracy scores : {accs_train}")

print(f"test accuracy scores : {accs_test}")


cv_res1 = pd.DataFrame.from_dict(model_svc.cv_results_)
cv_res2 = pd.DataFrame.from_dict(model_forest.cv_results_)

print("Ranking of the hyperparameter tuning for support vector classifier")
print(cv_res1.sort_values(by = "rank_test_score").head())

print("Ranking of the hyperparameter tuning for random forest classifier")
print(cv_res2.sort_values(by = "rank_test_score").head())

ConfusionMatrixDisplay.from_estimator(model_svc, X_test, y_test)

ConfusionMatrixDisplay.from_estimator(model_forest, X_test, y_test)

ConfusionMatrixDisplay.from_estimator(model_lreg, X_test, y_test)

print("Value_counts of y_test: \n", y_test.value_counts())

# Sampling
print("As bias is clearly shown, let's deal with the skewed target data problem using undersampling and oversampling.")
print("We choose Random_forest, as it is the best of the previous model.")

under_sampler = RandomUnderSampler()
X_train_under, y_train_under = under_sampler.fit_resample(X_train, y_train)
print("Shape of X_train after RandomUnderSampling")
print(X_train_under.shape)
X_train_under.head()

over_sampler = RandomOverSampler()
X_train_over, y_train_over = over_sampler.fit_resample(X_train, y_train)
print("Shape of X_train after RandomOverSampling")
print(X_train_over.shape)
X_train_over.head()

model_under = make_pipeline( 
    OneHotEncoder(use_cat_names = True),
    RandomForestClassifier(n_estimators = 50, max_depth = 10, random_state = 42)
)
model_under.fit(X_train_under, y_train_under)

model_over = make_pipeline( 
    OneHotEncoder(use_cat_names = True),
    RandomForestClassifier(n_estimators = 50, max_depth = 10, random_state = 42)
)
model_over.fit(X_train_over, y_train_over)


sampling_accs = []
options = [model_under, model_over]
for val in options: 
    sampling_accs.append(val.score(X_test, y_test))


print(f"accuracy scores after sampling: {sampling_accs}")


ConfusionMatrixDisplay.from_estimator(model_under, X_test, y_test)


ConfusionMatrixDisplay.from_estimator(model_over, X_test, y_test)

print("The best model is a random forest using random oversampling method")

# Get feature names from training data
features = model_over.named_steps["randomforestclassifier"].feature_names_in_
# Extract importances from model
importances = model_over.named_steps["randomforestclassifier"].feature_importances_
# Create a series with feature names and importances
feat_imp = pd.Series(importances, index = features)
# Plot the 10 most important features
feat_imp.sort_values().tail(10).plot(kind="barh")
plt.xlabel("Gini Importance")
plt.ylabel("Feature")
plt.title("Feature Importance");


with open("finalmodel", "wb") as f:
    pickle.dump(model_over, f)