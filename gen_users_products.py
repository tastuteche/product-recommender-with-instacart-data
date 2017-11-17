import pandas as pd
import numpy as np
b_dir = '../instacart/order_products__prior.csv/'
b1_dir = '../instacart/products.csv/'
b2_dir = '../instacart/orders.csv/'
order_products__prior = pd.read_csv(b_dir + 'order_products__prior.csv')
products = pd.read_csv(b1_dir + 'products.csv')
orders = pd.read_csv(b2_dir + 'orders.csv')
orders = orders.loc[orders['eval_set'] == 'prior']
order_prior = pd.merge(orders, order_products__prior, on=["order_id"])


def get_users_products(order_prior):
    users_products = order_prior. \
        groupby(["user_id", "product_id"]). \
        agg({'reordered': {'up_nb_reordered': "size"}})
    users_products.columns = users_products.columns.droplevel(0)
    users_products = users_products.reset_index()
    return users_products


users_products = get_users_products(order_prior)
users_products.to_csv('users_products.csv')
