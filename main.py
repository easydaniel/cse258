import gzip
import unicodedata
import seaborn as sns
import dask.dataframe as dd
import swifter
import string
import math
import pickle
import pandas as pd
import re
from tqdm import tqdm
import subprocess as sp
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
import numpy as np
from nltk.stem.porter import *
from nltk.stem import *
import nltk
from wordcloud import WordCloud
import matplotlib.pyplot as plt
tqdm.pandas()

DATASET = '/home/allenwhale/dataset.pickle.gz'
DATASET_DF = 'dataset.df.pickle'
DATASET_PROC_DF = 'dataset.proc.pickle'
CATEGORIES = 'categories.pickle'
# df = pickle.load(open(DATASET_DF, 'rb'))
# df['popular_shelves'] = df['popular_shelves'].progress_map(
#     lambda x: [{'name': s['name'].strip().lower(), 'count':s['count']} for s in x])


# gs = df.groupby('category')['popular_shelves']
# for c in gs.groups.keys():
#     g = gs.get_group(c)
#     count = defaultdict(int)
#     for s in g:
#         for x in s:
#             count[x['name']] += int(x['count'])
#     wc = WordCloud(background_color="white",
#                    max_words=500, width=1024, height=720)
#     wc.generate_from_frequencies(count)
#     plt.figure(figsize=(20, 10))
#     plt.imshow(wc, interpolation="bilinear")
#     plt.axis("off")
#     plt.savefig('shelves_%s.png' % c, format='png')

# dfs = defaultdict(int)
# for s in tqdm(df['popular_shelves']):
#     for x in s:
#         dfs[x['name']] += max(1, int(x['count']))
# idfs = {f: math.log10(len(df) / dfs[f]) for f in tqdm(dfs)}
# df = df[df['category'] == 'comics_graphic']
# gs = df.groupby('category')['popular_shelves']
# for c in gs.groups.keys():
#     g = gs.get_group(c)
#     count = defaultdict(int)
#     for s in g:
#         for x in s:
#             count[x['name']] += max(1, int(x['count']))
#     count = {x: count[x] * idfs[x] for x in count}
#     wc = WordCloud(background_color="white",
#                    max_words=500, width=1024, height=720)
#     wc.generate_from_frequencies(count)
#     plt.figure(figsize=(20, 10))
#     plt.imshow(wc, interpolation="bilinear")
#     plt.axis("off")
#     plt.savefig('shelves_tfidf_%s.png' % c, format='png')
#
#
# def clean3(x):
#     return '-'.join(filter(lambda x: len(x) >= 3, x.split('-')))
#
#
# df['popular_shelves_3'] = df['popular_shelves'].progress_map(
#     lambda x: [{'name': clean3(s['name']), 'count':s['count']} for s in x])
# gs = df.groupby('category')['popular_shelves_3']
# for c in gs.groups.keys():
#     g = gs.get_group(c)
#     count = defaultdict(int)
#     for s in g:
#         for x in s:
#             count[x['name']] += int(x['count'])
#     wc = WordCloud(background_color="white",
#                    max_words=500, width=1024, height=720)
#     wc.generate_from_frequencies(count)
#     plt.figure(figsize=(20, 10))
#     plt.imshow(wc, interpolation="bilinear")
#     plt.axis("off")
#     plt.savefig('shelves_3_%s.png' % c, format='png')
#
# snowball = SnowballStemmer('english')
# lemmatizer = WordNetLemmatizer()
#
#
# def clean_stem(x):
#     return '-'.join(snowball.stem(lemmatizer.lemmatize(s)) for s in x.split('-'))
#
#
# df['popular_shelves_stem'] = df['popular_shelves'].progress_map(
#     lambda x: [{'name': clean_stem(s['name']), 'count':s['count']} for s in x])
# gs = df.groupby('category')['popular_shelves_stem']
# for c in gs.groups.keys():
#     g = gs.get_group(c)
#     count = defaultdict(int)
#     for s in g:
#         for x in s:
#             count[x['name']] += int(x['count'])
#     wc = WordCloud(background_color="white",
#                    max_words=500, width=1024, height=720)
#     wc.generate_from_frequencies(count)
#     plt.figure(figsize=(20, 10))
#     plt.imshow(wc, interpolation="bilinear")
#     plt.axis("off")
#     plt.savefig('shelves_stem_%s.png' % c, format='png')
#
#
# exit()
# wc = WordCloud(background_color="white", max_words=500, width=1024, height=720)
# wc.generate_from_frequencies(wordCount)
# plt.figure(figsize=(20,10))
# plt.imshow(wc, interpolation="bilinear")
# plt.axis("off")
# plt.savefig(name, format = 'png')
# df = pd.DataFrame(pickle.load(gzip.open(DATASET)))
# df.to_pickle(DATASET_DF)
# df = pickle.load(open(DATASET_DF, 'rb'))
# pickle.dump(list(df['category'].unique()), open(CATEGORIES, 'wb'))
# PUNCTUATION_TRANS = str.maketrans('', '', string.punctuation)
# df = pickle.load(open(DATASET_DF, 'rb')).sample(frac=1.0)
# categories = pickle.load(open(CATEGORIES, 'rb'))
# df['label'] = df['category'].progress_apply(categories.index)
# snowball = SnowballStemmer('english')
# lemmatizer = WordNetLemmatizer()
# df['label'] = df['category'].progress_apply(categories.index)
# stopwords = set(nltk.corpus.stopwords.words('english'))
# df['ngrams'] = df['description'].progress_apply(
#     lambda x: list(
#         filter(
#             lambda x: len(x) >= 3 and x not in stopwords,
#             map(lambda x: snowball.stem(lemmatizer.lemmatize(x)),
#                 x.lower().translate(PUNCTUATION_TRANS).split())
#         )
#     )
# )
# df['ngrams_set'] = df['ngrams'].progress_apply(lambda x: set(x))
#
#
# def clean_shelves(x):
#     res = defaultdict(int)
#     for i in x:
#         k, v = i['name'], int(i['count'])
#         k = unicodedata.normalize('NFKD', k).encode('ASCII', 'ignore').decode()
#         k = re.sub('\s+', '', k.lower())
#         k = ''.join(
#             filter(lambda x: len(x) >= 3,
#                    (re.sub('\s+', '', snowball.stem(lemmatizer.lemmatize(x))) for x in k.split('-'))))
#         if len(k) < 3:
#             continue
#         res[k] += max(v, 0)
#     return dict(res)
#
#
# df['popular_shelves'] = df['popular_shelves'].progress_apply(clean_shelves)
# df.to_pickle('dataset.proc.pickle')
# exit()


def fasttext():
    FASTTEXT = '../fastText-0.1.0/fasttext'
    df = pickle.load(open(DATASET_DF, 'rb')).sample(frac=1.0)
    df['description'].str.replace('\s+', ' ')
    df = df[df['description'] != '']
    categories = pickle.load(open(CATEGORIES, 'rb'))
    train, test = df.iloc[:len(df) * 7 // 10], df.iloc[len(df) * 7 // 10:]
    with open('train.txt', 'w') as f:
        for _, x in tqdm(train.iterrows()):
            f.write('__label__%d %s\n' %
                    (categories.index(x['category']), re.sub('\s+', ' ', x['description'])))
    with open('test.txt', 'w') as f:
        for _, x in tqdm(test.iterrows()):
            f.write('__label__%d %s\n' %
                    (categories.index(x['category']), re.sub('\s+', ' ', x['description'])))
    for dim in range(100, 1001, 100):
        print('Running dim =', dim)
        sp.call([FASTTEXT, 'supervised', '-input', 'train.txt', '-output',
                 'model', '-dim', str(dim), '-epoch', '10', '-thread', '3'])
        sp.call([FASTTEXT, 'test', 'model.bin', 'test.txt'])
        sp.Popen([FASTTEXT, 'predict', 'model.bin', 'test.txt'],
                 stdout=open('fasttext_%d.pred' % dim, 'w')).communicate()


def fasttest_pr(dim):
    label = map(lambda x: int(re.findall('(\d+)', x.strip().split()[0].strip())[0]),
                open('test.txt', 'r').readlines())
    pred = map(lambda x: int(re.findall('(\d+)', x.strip())[0]),
               open('fasttext_%d.pred' % dim, 'r').readlines())
    label_pred = list(zip(label, pred))
    categories = pickle.load(open(CATEGORIES, 'rb'))
    for i in range(8):
        precision = len(list(filter(lambda x: x[0] == i and x[1] == i, label_pred))) / len(
            list(filter(lambda x: x[1] == i, label_pred)))
        recall = len(list(filter(lambda x: x[0] == i and x[1] == i, label_pred))) / len(
            list(filter(lambda x: x[0] == i, label_pred)))
        accuracy = (len(list(filter(lambda x: x[0] == i and x[1] == i, label_pred))) + len(
            list(filter(lambda x: x[0] != i and x[1] != i, label_pred)))) / len(label_pred)
        print(categories[i], 'precision:', precision)
        print(categories[i], 'recall:', recall)
        print(categories[i], 'accuracy:', accuracy)
    total_accuracy = len(
        list(filter(lambda x: x[0] == x[1], label_pred))) / len(label_pred)
    print('Total Accuracy:', total_accuracy)


def generate_description_features(gram_size, feature_size, feature_type):
    def gen_ngrams(s):
        res = []
        for i in range(len(s) - (gram_size - 1)):
            res.append(tuple(s[i:i + gram_size]))
        return res

    def idf(w, df):
        try:
            return math.log10(len(df) / len(df[df['ngrams_set'].apply(
                lambda x: w in x
            )]))
        except:
            return 0

    def gen_features(df, feature_keys):
        grams_count_each = []
        dfs = defaultdict(int)
        for r in tqdm(df['ngrams']):
            grams_count_each.append(defaultdict(int))
            for x in set(r) & set(feature_keys):
                grams_count_each[-1][x] += 1
            if feature_type == 'tfidf':
                for x in set(r) & set(feature_keys):
                    dfs[x] += 1
        if feature_type == 'tfidf':
            idfs = [math.log10(len(df) / dfs[f]) if dfs[f] !=
                    0 else 0 for f in tqdm(feature_keys)]
        features = []
        for i, r in enumerate(tqdm(df['ngrams'])):
            feature = np.zeros(1 + len(feature_keys))
            feature[-1] = 1
            for j, f in enumerate(feature_keys):
                if feature_type == 'count':
                    feature[j] = grams_count_each[i][f]
                elif feature_type == 'tfidf':
                    feature[j] = grams_count_each[i][f] * idfs[j]
            features.append(feature)
        features = np.array(features)
        return features
    stemmer = PorterStemmer()
    snowball = SnowballStemmer('english')
    lemmatizer = WordNetLemmatizer()
    stopwords = set(nltk.corpus.stopwords.words('english'))
    categories = pickle.load(open(CATEGORIES, 'rb'))
    df = pickle.load(open(DATASET_PROC_DF, 'rb')).sample(frac=1.0)
    df['ngrams'] = df['ngrams'].progress_apply(gen_ngrams)
    # df['label'] = df['category'].progress_apply(categories.index)
    # df['ngrams'] = df['description'].progress_apply(
    #     lambda x: gen_ngrams(
    #         list(
    #             filter(
    #                 lambda x: len(x) >= 3 and x not in stopwords,
    #                 map(lambda x: snowball.stem(lemmatizer.lemmatize(x)),
    #                     x.lower().translate(PUNCTUATION_TRANS).split())
    #             )
    #         )
    #     )
    # )
    # df['ngrams_set'] = df['ngrams'].progress_apply(lambda x: set(x))
    train, test = df.iloc[:len(df) * 7 // 10], df.iloc[len(df) * 7 // 10:]
    feature_keys = []
    words_dfs = defaultdict(int)
    for r in tqdm(train['ngrams']):
        for x in set(r):
            words_dfs[x] += 1
    words_idfs = {x: math.log10(
        len(train) / words_dfs[x]) if words_dfs[x] != 0 else 0 for x in tqdm(words_dfs)}

    words_list_each = []
    words_set_each = []
    for i in range(8):
        tmpdf = train[train['label'] == i]
        words_count = defaultdict(int)
        for r in tqdm(tmpdf['ngrams']):
            for x in r:
                words_count[x] += 1
        # words_idfs = {x: idf(x, tmpdf) for x in tqdm(words_set)}
        words_list = sorted(
            words_count.keys(), key=lambda x: -words_count[x] * words_idfs[x])[:feature_size // 8]
        print(words_list[:100])
        feature_keys += words_list
    feature_keys = list(set(feature_keys))
    #
    # grams_count = defaultdict(int)
    # for r in tqdm(train['ngrams']):
    #     for x in r:
    #         grams_count[x] += 1
    # grams_keys = sorted(grams_count.keys(), key=lambda x: -grams_count[x])
    # feature_keys = grams_keys[:feature_size]
    print('gen train features')
    train_X = gen_features(train, feature_keys)
    train_y = train['label'].values
    print('gen test features')
    test_X = gen_features(test, feature_keys)
    test_y = test['label'].values
    return train_X, train_y, test_X, test_y


def generate_shelves_features(df, feature_size, feature_type):
    categories = pickle.load(open(CATEGORIES, 'rb'))
    # df = pickle.load(open(DATASET_PROC_DF, 'rb')).sample(frac=1.0)
    # df['label'] = df['category'].progress_apply(categories.index)
    snowball = SnowballStemmer('english')
    lemmatizer = WordNetLemmatizer()

    def clean_shelves(x):
        res = defaultdict(int)
        for i in x:
            k, v = i['name'], int(i['count'])
            k = re.sub('\s+', '', k.lower())
            k = ''.join(
                filter(lambda x: len(x) >= 3,
                       (re.sub('\s+', '', snowball.stem(lemmatizer.lemmatize(x))) for x in k.split('-'))))
            if len(k) < 3:
                continue
            res[k] += max(v, 0)
        return dict(res)

    def gen_features(df, feature_keys):
        shelves_count_each = []
        if feature_type == 'tfidf':
            dfs = defaultdict(int)
        for s in tqdm(df['popular_shelves']):
            shelves_count_each.append(defaultdict(int))
            for k in set(s.keys()) & set(feature_keys):
                shelves_count_each[-1][k] += s[k]
            if feature_type == 'tfidf':
                for k in set(s.keys()) & set(feature_keys):
                    dfs[k] += 1
        if feature_type == 'tfidf':
            idfs = [math.log10(len(df) / dfs[f]) if dfs[f] !=
                    0 else 0 for f in tqdm(feature_keys)]
        features = []
        for i, r in enumerate(tqdm(df['popular_shelves'])):
            feature = np.zeros(len(feature_keys) + 1)
            feature[-1] = 1
            for j, f in enumerate(feature_keys):
                if feature_type == 'count':
                    feature[j] = shelves_count_each[i][f]
                elif feature_type == 'tfidf':
                    feature[j] = shelves_count_each[i][f] * idfs[j]
            features.append(feature)
        features = np.array(features)
        return features

    # df['popular_shelves'] = df['popular_shelves'].progress_apply(clean_shelves)
    train, test = df.iloc[:len(df) * 7 // 10], df.iloc[len(df) * 7 // 10:]
    shelves_dfs = defaultdict(int)
    for s in train['popular_shelves']:
        for x in s:
            shelves_dfs[x] += 1
    shelves_idfs = {x: math.log10(
        len(train) / shelves_dfs[x]) if shelves_dfs[x] != 0 else 0 for x in tqdm(shelves_dfs)}
    feature_keys = []
    for i in range(8):
        tmpdf = train[train['label'] == i]
        shelves_count = defaultdict(int)
        for s in tmpdf['popular_shelves']:
            for k, v in s.items():
                shelves_count[k] += v
        shelves_list = sorted(
            shelves_count.keys(), key=lambda x: -shelves_count[x] * shelves_idfs[x])[:feature_size // 8]
        print(shelves_list[:100])
        feature_keys += shelves_list
        del shelves_list
        del shelves_count
    feature_keys = list(set(feature_keys))
    print('gen train features')
    train_X = gen_features(train, feature_keys)
    train_y = train['label'].values
    print('gen test features')
    test_X = gen_features(test, feature_keys)
    test_y = test['label'].values
    return train_X, train_y, test_X, test_y


# x, y1, y2 = [1, 2], [3, 4], [5, 6]
# df = pd.DataFrame([{'x': x[i], 'type':'y1', 'y':y1[i]}for i in range(
#     len(x))] + [{'x': x[i], 'type':'y2', 'y':y2[i]}for i in range(len(x))])
# plot = sns.lineplot(x='x', y='y', data=df, hue='type')
# plot.get_figure().savefig('test.png')
# exit()
x, y1, y2 = [], [], []
df = pickle.load(open(DATASET_PROC_DF, 'rb')).sample(frac=1.0)
for dim in [8] + list(range(80, 2001, 80)):
    # train_X, train_y, test_X, test_y = generate_shelves_features(
    #     df, dim, 'tfidf')
    train_X, train_y, test_X, test_y = generate_description_features(
        1, dim, 'tfidf')
    clf = LogisticRegression(
        solver='lbfgs', multi_class='multinomial', n_jobs=3)
    clf.fit(train_X, train_y)
    x.append(dim)
    y1.append(clf.score(test_X, test_y))
    print('Log', dim, clf.score(test_X, test_y))
    # train_X, train_y, test_X, test_y = generate_shelves_features(
    #     df, dim, 'count')
    train_X, train_y, test_X, test_y = generate_description_features(
        1, dim, 'count')
    clf = LogisticRegression(
        solver='lbfgs', multi_class='multinomial', n_jobs=3)
    clf.fit(train_X, train_y)
    y2.append(clf.score(test_X, test_y))
    print('Log', dim, clf.score(test_X, test_y))
d = []
d += [{'Features': j, 'type': 'tfidf', 'Accuracy': y1[i]}
      for i, j in enumerate(x)]
d += [{'Features': j, 'type': 'count', 'Accuracy': y2[i]}
      for i, j in enumerate(x)]
plot = sns.lineplot(x='Features', y='Accuracy',
                    data=pd.DataFrame(d), hue='type')
plot.get_figure().savefig('log_description_features2acc_8_2000_80.png')


# btrain_X, train_y, btest_X, test_y = generate_description_features(
#     1, 1000, 'tfidf')
train_X, train_y, test_X, test_y = generate_shelves_features(2000, 'tfidf')
# exit()
# train_X, train_y, test_X, test_y = generate_description_features(
#     1, 4000, 'count')
# pickle.dump((train_X, train_y, test_X, test_y), open('features.pickle', 'wb'))
# clf = AdaBoostClassifier(RandomForestClassifier(), n_estimators=10)
# clf.fit(train_X, train_y)
# print('Ada + RF', clf.score(test_X, test_y))
clf = DummyClassifier()
clf.fit(train_X, train_y)
print('Dummy', clf.score(test_X, test_y))
clf = LogisticRegression(
    solver='lbfgs', multi_class='multinomial', n_jobs=3)
clf.fit(train_X, train_y)
print('Log', clf.score(test_X, test_y))
clf = MultinomialNB()
clf.fit(train_X, train_y)
print('NB', clf.score(test_X, test_y))
clf = RandomForestClassifier()
clf.fit(train_X, train_y)
print('RF', clf.score(test_X, test_y))
clf = DecisionTreeClassifier()
clf.fit(train_X, train_y)
print('DT', clf.score(test_X, test_y))
clf = VotingClassifier(estimators=[
    ('log', LogisticRegression(
        solver='lbfgs', multi_class='multinomial')),
    ('NB', MultinomialNB()),
    ('RF', RandomForestClassifier()),
    ('DT', DecisionTreeClassifier())
], n_jobs=3)
clf.fit(train_X, train_y)
print('voting', clf.score(test_X, test_y))
# clf = GaussianNB()
# clf.fit(train_X, train_y)
# print('GNB', clf.score(test_X, test_y))
# clf = KNeighborsClassifier(n_jobs=3)
# clf.fit(train_X, train_y)
# print('KNN', clf.score(test_X, test_y))
# exit()
# clf = AdaBoostClassifier(DecisionTreeClassifier(), n_estimators=10)
# clf.fit(train_X, train_y)
# print('Ada + DT', clf.score(test_X, test_y))
# clf = SVC()
# clf.fit(train_X, train_y)
# print('SVC', clf.score(test_X, test_y))
# clf = AdaBoostClassifier(SVC(), n_estimators=10)
# clf.fit(train_X, train_y)
# print('Ada + SVC', clf.score(test_X, test_y))
# fasttest_pr(400)
# fasttext()
