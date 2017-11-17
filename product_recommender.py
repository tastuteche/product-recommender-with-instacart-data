import pandas as pd
import numpy as np

b1_dir = '../instacart/products.csv/'
products = pd.read_csv(b1_dir + 'products.csv')
dic_product = products.set_index('product_id')['product_name'].to_dict()
users_products = pd.read_csv('users_products.csv')

max_count = users_products['up_nb_reordered'].max()
# max_count = 100

from surprise import SVD
from surprise import Dataset, Reader
from surprise import evaluate, print_perf

# reader = Reader(line_format="user item rating", sep=',',
#                 rating_scale=(1, max_count), skip_lines=1)
# data = Dataset.load_from_file('users_products.csv', reader=reader)
# df_001 = df.sample(frac=0.0001)
# df_rest = df.loc[~df.index.isin(df_001.index)]

df_train = users_products.sample(frac=0.0001)
df_rest = users_products.loc[~users_products.index.isin(df_train.index)]

reader = Reader(rating_scale=(1, max_count))
data = Dataset.load_from_df(
    df_train[['user_id', 'product_id', 'up_nb_reordered']], reader)
data.split(n_folds=3)

# We'll use the famous SVD algorithm.
algo = SVD()

# Evaluate performances of our algorithm on the dataset.
perf = evaluate(algo, data, measures=['RMSE', 'MAE'])

print_perf(perf)
train_set = data.build_full_trainset()
train = Dataset.load_from_df(
    df_train[['user_id', 'product_id', 'up_nb_reordered']], reader)
algo.train(train_set)
uid = str(196)  # raw user id
iid = str(302)  # raw item id
# get a prediction for specific users and items.
pred = algo.predict(uid, iid)

#results_df = pd.DataFrame.from_dict(algo.cv_results)
# print(results_df)

from collections import defaultdict


def get_top_n(predictions, n=10):
    '''Return the top-N recommendation for each user from a set of predictions.
    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.
    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    '''
    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))
    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]
    return top_n



# Than predict ratings for all pairs (u, i) that are NOT in the training set.
test_set = train_set.build_anti_testset()
predictions = algo.test(test_set)

top_n = get_top_n(predictions, n=5)


def get_count(uid, iid):
    return len(users_products[users_products['user_id'].isin([uid]) & users_products['product_id'].isin([iid])])


# Print the recommended items for each user
for uid, user_ratings in top_n.items():
    # print(uid, [(dic_product[iid], get_count(uid, iid))
    print(uid, [dic_product[iid] for (iid, _) in user_ratings])
