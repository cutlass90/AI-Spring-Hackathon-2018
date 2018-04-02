from user_agents import parse
import pandas as pd
from dateutil import parser
from polyglot.text import Text
from multiprocessing import Pool
import numpy as np
from polyglot.detect import Detector

csv_train = '../data/train.csv'
csv_test = '../data/test.csv'

def apply(func, list_):
    with Pool(8) as pool:
        res = pool.map(func, list_)
    return pd.Series(res)


def foo(x):
    try:
        lang = Detector(x).language.name
        if lang not in ['Russian', 'Ukrainian']:
            lang = 'unk'

    except:
        lang = 'unk'
    return lang


class DataProvider:

    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        s = self.df.shape[0]
        print('Found {} samples'.format(s))
        self.df.product_id.fillna(1005004242, inplace=True)
        self.df = self.df.dropna()
        print('Delete {} samples with NAN'.format(s - self.df.shape[0]))
        self.unk_code_int = 100500
        self.fit()

    def one_hot(self, uniq, series, unk_code):
        ser = series.apply(lambda x: x if x in uniq else unk_code)
        return pd.get_dummies(ser)

    def fit(self):
        # self.company_id_uniq = self.df[self.df.is_fake==True]['company_id'].unique()
        # self.user_id_uniq = self.df[self.df.is_fake == True]['user_id'].unique()
        # self.product_id_uniq = self.df[self.df.is_fake == True]['product_id'].unique()
        self.df['date_created_unix'] = apply(str2unix, self.df.date_created.tolist())
        self.df['user_date_created_unix'] = apply(str2unix, self.df.user_date_created.tolist())
        self.maxdatetime_diff = (self.df.date_created_unix - self.df.user_date_created_unix).max()
        self.date_created_unix_max = self.df['date_created_unix'].max()
        self.user_date_created_unix_max = self.df['user_date_created_unix'].max()

        self.company_id_mean = self.df.groupby(['company_id']).is_fake.mean()
        self.company_id_mean = {id: self.company_id_mean[id] for id in self.company_id_mean.index}
        
        self.company_id_sum = self.df.groupby(['company_id']).is_fake.sum()
        self.company_id_sum = {id: self.company_id_sum[id] for id in self.company_id_sum.index}
        self.company_id_sum_max = max([i for i in list(self.company_id_sum.values())])

        self.user_id_mean = self.df.groupby(['user_id']).is_fake.mean()
        self.user_id_mean = {id: self.user_id_mean[id] for id in self.user_id_mean.index}

        self.user_id_sum = self.df.groupby(['user_id']).is_fake.sum()
        self.user_id_sum = {id: self.user_id_sum[id] for id in self.user_id_sum.index}
        self.user_id_sum_max = max([i for i in list(self.user_id_sum.values())])


        self.product_id_mean = self.df.groupby(['product_id']).is_fake.mean()
        self.product_id_mean = {id: self.product_id_mean[id] for id in self.product_id_mean.index}

        self.product_id_sum = self.df.groupby(['product_id']).is_fake.sum()
        self.product_id_sum = {id: self.product_id_sum[id] for id in self.product_id_sum.index}
        self.product_id_sum_max = max([i for i in list(self.product_id_sum.values())])

        print('fit finished')

    def transform(self, csv_path=None):
        df = self.df if csv_path == None else pd.read_csv(csv_path)
        if csv_path is not None:
            df.user_agent.fillna('', inplace=True)
        parsed_user_agent = apply(parse, df.user_agent.tolist())

        # company_id_onehot = self.one_hot(self.company_id_uniq, df.company_id, unk_code=self.unk_code_int)
        # print('company_id_onehot')
        # user_id_onehot = self.one_hot(self.user_id_uniq, df.user_id, unk_code=self.unk_code_int)
        # print('user_id_onehot')
        # product_id_onehot = self.one_hot(self.product_id_uniq, df.product_id, unk_code=self.unk_code_int)
        # print('product_id_onehot')
        rating_onehot = pd.get_dummies(df.rating)
        print('rating_onehot')
        if csv_path is not None:
            df['date_created_unix'] = apply(str2unix, df.date_created.tolist())
            df['user_date_created_unix'] = apply(str2unix, df.user_date_created.tolist())
        datetime_diff = (df.date_created_unix - df.user_date_created_unix)/self.maxdatetime_diff
        date_created_unix = df['date_created_unix']/self.date_created_unix_max
        user_date_created_unix = df['user_date_created_unix'] / self.user_date_created_unix_max

        is_mobile = parsed_user_agent.apply(lambda x: x.is_mobile)
        is_touch_capable = parsed_user_agent.apply(lambda x: x.is_touch_capable)
        is_pc = parsed_user_agent.apply(lambda x: x.is_pc)
        is_bot = parsed_user_agent.apply(lambda x: x.is_bot)

        company_id_mean = df.company_id.apply(lambda x: self.company_id_mean.get(x, 0))
        company_id_sum = df.company_id.apply(lambda x: self.company_id_sum.get(x, 0))
        company_id_sum /= self.company_id_sum_max
        user_id_mean = df.user_id.apply(lambda x: self.user_id_mean.get(x, 0))
        user_id_sum = df.user_id.apply(lambda x: self.user_id_sum.get(x, 0))
        user_id_sum /= self.user_id_sum_max
        product_id_mean = df.product_id.apply(lambda x: self.product_id_mean.get(x, 0))
        product_id_sum = df.product_id.apply(lambda x: self.product_id_sum.get(x, 0))
        product_id_sum /= self.product_id_sum_max
        languages = pd.get_dummies(apply(foo, df.comment.tolist()))

        to_concat = [
            # company_id_onehot,
            # user_id_onehot,
            # product_id_onehot,
            rating_onehot,
            datetime_diff,
            date_created_unix,
            user_date_created_unix,
            is_mobile,
            is_touch_capable,
            is_pc,
            is_bot,
            company_id_mean,
            company_id_sum,
            user_id_mean,
            user_id_sum,
            product_id_mean,
            product_id_sum,
            languages
        ]

        print('shapes')
        for d in to_concat:
            print(d.shape)

        X = pd.concat(to_concat, axis=1, ignore_index=True)
        print(X.shape)
        X.fillna(0, inplace=True)
        print('transform finished')
        return X


def str2unix(text):
    return parser.parse(text).timestamp()


if __name__ == '__main__':
    data_provider = DataProvider(csv_train)
    Y = data_provider.df.is_fake
    Y.to_pickle('../Y.pkl')
    X_train = data_provider.transform()
    X_train.to_pickle('../X_train.pkl')
    X_test = data_provider.transform(csv_test)
    X_test.to_pickle('../X_test.pkl')



