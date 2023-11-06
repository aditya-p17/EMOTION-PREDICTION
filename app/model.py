

import numpy as np
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from collections import Counter
# pip install alt-profanity-check
# pip install sklearn --upgrade
# pip install joblib
# import xgboost
import warnings
warnings.filterwarnings('ignore')

dataset=pd.read_csv("Data for AI Assignment - Sheet1.csv")
# dataset.head()

dataset.info()

classes=set(dataset["Classification"])
print(classes)
data_number=dict(Counter(dataset["Classification"]))
# data_number

import matplotlib.pyplot as plt

names = list(data_number.keys())
values = list(data_number.values())

plt.bar(range(len(data_number)), values, tick_label=names)
plt.show()

duplicate_mask = dataset.duplicated(subset=['Text'], keep='first')

# Select duplicate rows
duplicate_rows = dataset[duplicate_mask]

# print("Duplicate Rows:")
# display(duplicate_rows)

dataset = dataset.drop_duplicates(subset=['Text'], keep='first')
# dataset

# import sklearn
from profanity_check import predict, predict_prob

#predict_prob(["ive been taking or milligrams or times recommended amount and ive fallen asleep a lot faster but i also feel like so funny","i feel pretty pathetic most of the time","i am not a people person but for some fuckin reason people feel that they can come bore me with their fuckin petty garbage","i feel slightly disgusted as well","i feel alarmed","you piece of shit"])

dataset[predict_prob(dataset["Text"])>=0.80]

dataset=dataset[predict_prob(dataset["Text"])<0.80]
dataset

dataset.info()

classes=set(dataset["Classification"])
print(classes)
data_number=dict(Counter(dataset["Classification"]))
# data_number

names = list(data_number.keys())
values = list(data_number.values())

plt.bar(range(len(data_number)), values, tick_label=names)
plt.show()

"""## DataAugumentation

"""

# pip install nlpaug

df_new=pd.DataFrame({"Text":[],"Classification":[]})

df=dataset[(dataset.Classification=="love")]
# df

import nlpaug
import nlpaug.augmenter.word as naw
aug = naw.SynonymAug(aug_src='wordnet',aug_max=2)
for i in dataset[(dataset.Classification=="love")].index:
  text=dataset["Text"][i]
  new_array=aug.augment(text,n=2)
  df_adder=pd.DataFrame({"Text":new_array[0],"Classification":"love"},{"Text":new_array[1],"Classification":"love"})
  df_new=pd.concat([df_adder,df_new])




df1=dataset[(dataset.Classification=="surprise")]


df_new1=pd.DataFrame({"Text":[],"Classification":[]})

import nlpaug
import nlpaug.augmenter.word as naw
aug = naw.SynonymAug(aug_src='wordnet',aug_max=2)
for i in dataset[(dataset.Classification=="surprise")].index:
  text=dataset["Text"][i]
  new_array=aug.augment(text,n=2)
  df_adder=pd.DataFrame({"Text":new_array[0],"Classification":"surprise"},{"Text":new_array[1],"Classification":"surprise"})
  df_new=pd.concat([df_adder,df_new])


print(df_new.info())
print(df_new1.info())

df_augment=pd.concat([df_new,df_new1])
df_augment= df_augment.sample(frac = 1)
# df_augment.head()

"""## Preprocessing"""

import string
from nltk.corpus import stopwords
nltk.download("stopwords")

dataset["Text"]=dataset["Text"].str.lower()
df_augment["Text"]=df_augment["Text"].str.lower()
# dataset.head()

def remove_punctuations(text):
    punctuations=string.punctuation
    return text.translate(str.maketrans("","",punctuations))

dataset["Text"]=dataset["Text"].apply(lambda x: remove_punctuations(x))
df_augment["Text"]=df_augment["Text"].apply(lambda x: remove_punctuations(x))
# dataset.head()

stopWords=set(stopwords.words("english"))
def remove_stopwords(text):
    return " ".join([word for word in text.split() if word not in stopWords])

dataset["Text"]=dataset["Text"].apply(lambda x: remove_stopwords(x))
df_augment["Text"]=df_augment["Text"].apply(lambda x: remove_stopwords(x))

from collections import Counter
word_count = Counter()
for text in dataset['Text']:
    for word in text.split():
        word_count[word] += 1

# word_count.most_common(10)

FREQUENT_WORDS = set(word for (word, wc) in word_count.most_common(10))
def remove_freq_words(text):
    return " ".join([word for word in text.split() if word not in FREQUENT_WORDS])

dataset['Text'] = dataset['Text'].apply(lambda x: remove_freq_words(x))
df_augment["Text"]=df_augment["Text"].apply(lambda x: remove_freq_words(x))
# dataset.head()

RARE_WORDS = set(word for (word, wc) in word_count.most_common()[:-10:-1])
# RARE_WORDS

def remove_rare_words(text):
    return " ".join([word for word in text.split() if word not in RARE_WORDS])
dataset['Text'] = dataset['Text'].apply(lambda x: remove_rare_words(x))
df_augment["Text"]=df_augment["Text"].apply(lambda x: remove_rare_words(x))
# dataset.head(15)


import re
def remove_spl_chars(text):
    text = re.sub('[^a-zA-Z0-9]', ' ', text)
    # text = re.sub('\s+', ' ', text)
    return text

dataset['Text'] = dataset['Text'].apply(lambda x: remove_spl_chars(x))
df_augment["Text"]=df_augment["Text"].apply(lambda x: remove_spl_chars(x))
# dataset.head()

# from nltk.stem.porter import PorterStemmer
# ps = PorterStemmer()
# def stem_words(text):
#     return " ".join([ps.stem(word) for word in text.split()])

# dataset['stemmed_text'] = dataset['Text'].apply(lambda x: stem_words(x))
# dataset.head()

nltk.download('wordnet')

nltk.download('averaged_perceptron_tagger')

from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
wordnet_map = {"N":wordnet.NOUN, "V": wordnet.VERB, "J": wordnet.ADJ, "R": wordnet.ADV}

def lemmatize_words(text):
    # find pos tags
    pos_text = pos_tag(text.split())
    return " ".join([lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_text])

#wordnet.NOUN //'n'
dataset['lemmatized_text'] = dataset['Text'].apply(lambda x: lemmatize_words(x))
df_augment["lemmatized_text"]=df_augment["Text"].apply(lambda x: lemmatize_words(x))
# dataset.head(15)

# df_augment.head()

dataset=dataset[["lemmatized_text","Classification"]]

# dataset.head(15)


df_augment=df_augment[["lemmatized_text","Classification"]]

"""## Feature Selection and Model-Training"""

possible_labels = dataset.Classification.unique()

label_dict = {}
for index, possible_label in enumerate(possible_labels):
    label_dict[possible_label] = index
label_dict
print(label_dict)

dataset["Classification"]=dataset.Classification.replace(label_dict)
df_augment["Classification"]=df_augment.Classification.replace(label_dict)
# dataset.head(15)

new_dataset=pd.concat([df_augment,dataset])
new_dataset.info()

new_dataset.Classification.unique()

classes=set(new_dataset["Classification"])
print(classes)
data_number=dict(Counter(new_dataset["Classification"]))
print(data_number)

shuffled_df=new_dataset.sample(frac=1).reset_index(drop=True)

from sklearn.model_selection import train_test_split
X =shuffled_df["lemmatized_text"]
y=shuffled_df["Classification"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state = 42)

# X_train.shape,X_test.shape,y_train.shape,y_test.shape

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer

lr = Pipeline([('vect', CountVectorizer()),
               ('tfidf', TfidfTransformer()),
               ('clf', LogisticRegression()),
              ])

lr.fit(X_train,y_train)

# acc_lr=accuracy_score(y_pred_lr,y_test)
# f1_score_lr=f1_score(y_pred_lr,y_test,average=None)
# f1_score_lr_weighted=f1_score(y_pred_lr,y_test,average="weighted")
# train_score_lr=lr.score(X_train,y_train)

# from sklearn.naive_bayes import MultinomialNB


# naivebayes = Pipeline([('vect', CountVectorizer()),
#                ('tfidf', TfidfTransformer()),
#                ('clf', MultinomialNB()),
#               ])
# naivebayes.fit(X_train, y_train)

# y_pred_nb = naivebayes.predict(X_test)

# acc_nb=accuracy_score(y_pred_nb,y_test)
# f1_score_nb=f1_score(y_pred_nb,y_test,average=None)
# f1_score_nb_weighted=f1_score(y_pred_nb,y_test,average="weighted")
# train_score_nb=naivebayes.score(X_train,y_train)

# from xgboost import XGBClassifier
# from sklearn.metrics import f1_score
# xgboost = Pipeline([('vect', CountVectorizer()),
#                ('tfidf', TfidfTransformer()),
#                ('clf', XGBClassifier()),
#               ])
# xgboost.fit(X_train, y_train)

# y_pred_xb = xgboost.predict(X_test)

# acc_xb=accuracy_score(y_pred_xb,y_test)
# f1_score_xb=f1_score(y_pred_xb,y_test,average=None)
# f1_score_xb_weighted=f1_score(y_pred_xb,y_test,average="weighted")
# train_score_xb=xgboost.score(X_train,y_train)

# from catboost import CatBoostClassifier
# params = {'learning_rate': 0.1, 'depth': 6,
#           'l2_leaf_reg': 3, 'iterations': 100}
# catboost = Pipeline([('vect', CountVectorizer()),
#                ('tfidf', TfidfTransformer()),
#                ('clf', CatBoostClassifier(**params)),
#               ])

# catboost.fit(X_train, y_train)
# y_pred_cb = catboost.predict(X_test)

# acc_cb=accuracy_score(y_pred_cb,y_test)
# f1_score_cb=f1_score(y_pred_cb,y_test,average=None)
# f1_score_cb_weighted=f1_score(y_pred_cb,y_test,average="weighted")

# analysis={"Model":["Logistic Regression","Naive Bayes","XGBoost"],
#           "Train Score":[train_score_lr,train_score_nb,train_score_xb],
#     "Accuracy_Score":[acc_lr,acc_nb,acc_xb],
#  "F1_Score":[f1_score_lr,f1_score_nb,f1_score_xb]
#           ,"F1_Score_weighted":[f1_score_lr_weighted,f1_score_nb_weighted,f1_score_xb_weighted]}
# df=pd.DataFrame(analysis)

# df

import pickle
pickle.dump(lr, open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))