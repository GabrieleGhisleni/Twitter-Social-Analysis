from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.snowball import ItalianStemmer
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from typing import List
import geopandas as gpd
import seaborn as sns
import datetime as time
import unicodedata
import pandas as pd
import numpy as np
import json, re


class NltkTextProcessing:
    def __init__(self):
        self.stemmer = ItalianStemmer(ignore_stopwords=True)
        self.stopwords = set(stopwords.words("italian"))
        self.increase_stopwords()

    def tokenize_text(self, text, stem: bool = False, min_len: int = 3) -> List or None:
        tokenized, res = word_tokenize(text=text, language='it'), list()
        for token in tokenized:
            if token not in self.stopwords and len(token) > 2 and not token.startswith('AH') and not token.startswith('TW'):
                if token.lower() == 'vaccini': token = 'vaccino'.upper()
                if token.lower() == 'vaccinati': token = 'vaccinato'.upper()
                if token.lower() == 'manifestazioni': token='manifestazione'.upper()
                res.append(token)
        if len(res) < min_len:
            return None
        if stem: res = [self.stemmer.stem(word) for word in res]
        return res

    def clean_text(self, text: str) -> str:
        def remove_emojy(text):
            regrex_pattern=re.compile(pattern="["
                                              u"\U0001F600-\U0001F64F"
                                              u"\U0001F300-\U0001F5FF"
                                              u"\U0001F680-\U0001F6FF"
                                              u"\U0001F1E0-\U0001F1FF"
                                              "]+", flags=re.UNICODE)
            return regrex_pattern.sub(r'', text).replace('\n', ' ')
        text = remove_emojy(text)
        for word in text.split(' '):
            if word.startswith('http'): text=text.replace(word, '')
        text = unicodedata.normalize('NFD', text)
        text = text.encode('ascii', 'ignore')
        text = text.decode("utf-8")
        text = text.replace("\n", '')
        text = text.upper()
        text = re.sub('[0-9$%]', ' ', text)
        text = re.sub("[^a-zA-Z;@#]+", ' ', text)
        for iel in range(4, 1, -1): text = text.replace(' ' * iel, ' ')
        text = text.replace('  ', ' ')
        text = text.strip()
        return text

    def process_df_text_column(self, df: pd.DataFrame, stem: bool, min_len: int = 3) -> pd.DataFrame:
        df.loc[:, ['tweet_text']]=df['tweet_text'].apply(self.clean_text)
        df.loc[:, ['tweet_text']] = df['tweet_text'].apply(self.tokenize_text, stem=stem, min_len=min_len)
        return df[df['tweet_text'].notna()]

    def take_only_keywords_from_tweets(self, tweets: pd.Series, sets, min_len: int = 1, stem: bool = False):
        tokens, final = tweets.apply(self.tokenize_text, stem=stem, min_len=1), list()
        for t in tokens.apply(NltkTextProcessing.check_key, res=sets).dropna():
            if len(t) > min_len: final.append(t)
        print(f"Remained docs: {len(final)}")
        return final

    def unique_hashtags(self, df: pd.DataFrame):
        wl=set()
        for tweets in np.unique(df['hashtags'].dropna()):
            for hashs in tweets:
                if len(hashs) > 3: wl.add(hashs)
        return wl

    def increase_stopwords(self) -> None:
        stopwords_ = {'ce', 'fa', 'tanto', 'comunque', 'ecco', 'sempre', 'perche', 'va', 'co', 't', 'vuole',
                    'dopo', 'https', 'poi', 'vedere', 'te', 'quest', 'do', 'no', 'pero', 'piu', 'quando',
                    'adesso', 'ogni', 'so', 'essere', 'tutta', 'senza', 'fatto', 'essere', 'oggi', 'cazzi', 'posso',
                    'altri', 'ah', 'quindi', 'gran', 'solo', 'ora', 'grazie', 'cosa', 'gia', 'me', '-', 'puoi',
                    'altro', 'nome', 'prima', 'anno', 'pure', 'qui', 'fate', 'sara', 'proprio', 'sa', 'de', 'fare',
                    'nuova', 'molto', 'mette', 'dire', 'tali', 'puo', 'uso', 'cioe', 'alta', 'far', 'qualsiasi',
                    'cosi', 'chiamano', 'capito', 'cazzo', 'raga', 'mai', 'avere', 'andare', 'invece', 'mesi', 'ancora',
                    'invece', 'a0xlp74lne', 'a4otny4rhy', 'aaa', 'aacmgmzanzio', 'aanzibma3f', 'ajgsd0w7mx', 'parli',
                    'vai','allegri', 'qusta', 'qusto', 'anch', 'prch', 'com', 'snza', 'dir', 'qlli', 'no', 'detto','dice',
                    'qualcuno','qualche','quali', 'ieri','oggi', 'ile','cio','altra','via','ilpass','delpass',
                    'quasi','die','andra','alle','https', 'luc','asono' ,'devo','avra','nun','non', 'accounthttps','ecc'
                    ,'sti','qua','neanche','oltre','vuol','chissa','roma','torino','milano','ancona','nizza','catania','agosto',
                    'settembre','alessandro','nulla','bene','sabato','domenica','napoli', 'MATTEOSALVINIMI','pochi','anni','molti',
                    'mica', 'rincoglionito','due','tre'}
        self.stopwords = self.stopwords.union(stopwords_)
        self.stopwords = self.stopwords.union(set(map(str.upper, self.stopwords)))

    def get_location(self, df: pd.DataFrame, path: str='data/citta.json') -> pd.DataFrame:
        with open(path, 'r') as file:
            location = json.load(file)
        region = set(location.values())
        punct = ['-','_','/',',','.','!','?']
        flag,res = True, list()
        for row in df.author_loc.dropna():
            for p in punct:
                tmp = row.replace(p, ' ')
            for word in tmp.split(' '):
                if word in location:
                    res.append(location[word])
                    flag = False
            if flag:
                for word in tmp.split(' '):
                    if word in region:
                        res.append(word)
            flag = True
        name, count = np.unique(res, return_counts=True)
        loc = dict(zip(name, count))
        loc = pd.DataFrame(loc, index=[0]).T.reset_index().rename(columns={'index': 'reg_name', 0: 'value'})
        loc.loc[loc.reg_name == 'Trentino-Alto Adige/S�dtirol', ['reg_name']] = 'Trentino-Alto Adige'
        return loc

    def get_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        def rename_month(txt):
            if txt[0:2] == '08': return f"{txt[3:]} Aug"
            else: return f"{txt[3:]} Sept"
        date, value = np.unique(df.created_at.apply(lambda x: x[5:10]).tolist(), return_counts=True)
        res = dict(zip(date, value))
        tmp = pd.DataFrame(res, index=[0]).T.reset_index().rename(columns={'index':'date',0:'value'})
        tmp.loc[:, 'date'] = tmp.loc[:, 'date'].apply(rename_month)
        return tmp

    def get_followers(self, df: pd.DataFrame) -> list:
        return sorted(df.author_followers.loc[df.author_followers > 1].tolist())

    def process_location(self, loc: pd.DataFrame) -> pd.DataFrame:
        italy_path="https://raw.githubusercontent.com/openpolis/geojson-italy/master/geojson/limits_IT_regions.geojson"
        italy=gpd.read_file(italy_path)
        italy.loc[italy.reg_name == 'Trentino-Alto Adige/Südtirol', ["reg_name"]] = 'Trentino-Alto Adige'
        return pd.merge(italy, loc, on='reg_name')

    def plot_dates_location_followers(self, df: pd.DataFrame, save: bool = False, path: str='data/citta.json') -> None:
        fig, axes=plt.subplots(1, 3, figsize=(18, 7))
        dates = self.get_dates(df)
        follower = self.get_followers(df)
        location = self.get_location(df, path)
        regions_df = self.process_location(location)
        sns.barplot(ax=axes[0], data=dates, y='date', x='value', palette='viridis')
        sns.lineplot(ax=axes[2], y=follower, x=[iel for iel in range(len(follower))], linewidth=5, color='firebrick')
        regions_df.plot(ax=axes[1],
                    column='value',
                    linewidth=0.1,
                    scheme='NaturalBreaks',
                    k=5,
                    legend=True,
                    markersize=45,
                    legend_kwds=dict(fmt='{:.0f}', frameon=False, loc='best'))
        axes[2].set_ylabel('Followers')
        axes[2].set_title('Users-Followers distribution')
        axes[2].set_xticks([])
        axes[0].tick_params(labelrotation=0)
        axes[0].set_title('Daily Rate')
        axes[1].set_title('Location Distribution')
        axes[0].set_xlabel('')
        axes[2].set_xlabel("User's index")
        axes[1].set_xticks([])
        axes[1].set_yticks([])
        axes[2].yaxis.tick_right()
        fig.tight_layout()
        if save:
            plt.savefig('photos/dataset.eps', format='eps', dpi=300)
        plt.show()

    @staticmethod
    def process_df_hash_column(df: pd.DataFrame, save: bool = False) -> pd.DataFrame:
        def hash_process(hashes):
            if hashes: return [hashs.lower() for hashs in hashes]
        df.loc[:, ['hashtags']] = df['hashtags'].apply(hash_process)
        if save: df.to_csv('tweets.csv')
        return df

    @staticmethod
    def keep_unique(df: pd.DataFrame, save: bool = False) -> pd.DataFrame:
        check, res = set(), list()
        for (idx, row) in df.iterrows():
            if tuple(row.tweet_text) in check:
                pass
            else:
                check.add(tuple(row.tweet_text))
                res.append(row)
        unique_df = pd.DataFrame(res)
        if save: unique_df.to_csv('unique.csv')
        return unique_df

    @staticmethod
    def frequency_dist(df: pd.DataFrame, obj: str = 'tweet') -> FreqDist:
        res = FreqDist()
        if obj=='tweet': bag= df['tweet_text']
        elif obj =='hash': bag = df['hashtags']
        else: bag = df
        for text in bag:
            if text:
                for word in text: res[word] += 1
        return res

    @staticmethod
    def extract_external_url(df: pd.DataFrame) -> dict:
        res = dict()
        for url in df['external_url']:
            if url and 'twitter' not in url:
                slash_idx = url.find('/') + 2
                idx = url.find(".")
                if 'www' in url:
                    sdx = url.find('.', idx + 1)
                    web = url[slash_idx:sdx + 4]
                else:
                    second_slash = url.find('/', slash_idx) + 1
                    web = url[slash_idx:second_slash]
                if web != '':
                    if web in res:
                        res[web] += 1
                    else: res[web] = 1
        return res

    @staticmethod
    def prepare_text_to_vectorize(df: pd.DataFrame, obj = 'tweet') -> list:
        if obj == 'tweet':
            return df['tweet_text'].apply(lambda x: ''.join(x)).values.tolist()
        else: return df['hashtags'].apply(lambda x: ''.join(x)).values.tolist()

    @staticmethod
    def check_key(s, res):
        ret=set()
        for val in s:
            if val in res: ret.add(val)
        if ret: return list(ret)
        else: return None


def update_parameter() -> None:
    large, med = 22, 16
    sns.set_style('white')
    params={'axes.titlesize': large,'legend.fontsize': med,
            'axes.labelsize': med, 'xtick.labelsize': large,
            'ytick.labelsize': large, 'figure.titlesize': large}
    plt.rcParams.update(params)
    sns.set_style('whitegrid')

