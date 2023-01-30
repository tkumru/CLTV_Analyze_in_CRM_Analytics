import pandas as pd

import datetime as dt                          

from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%3f' % x)

df_ = pd.read_csv("datasets/flo_data_20k.csv")
df = df_.copy()
                      
def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    
    return round(low_limit), round(up_limit)

def replace_with_threshold(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit
    
replace_with_threshold(df, "order_num_total_ever_online")
replace_with_threshold(df, "order_num_total_ever_offline")
replace_with_threshold(df, "customer_value_total_ever_offline")
replace_with_threshold(df, "customer_value_total_ever_online")

df["total_order_count"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["total_price"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]

def change_date(dataframe, variable):
    dataframe[variable] = dataframe[variable].astype("datetime64[ns]")
    
change_date(df, "first_order_date")
change_date(df, "last_order_date")
change_date(df, "last_order_date_online")
change_date(df, "last_order_date_offline")

last_date = df["last_order_date"].max() # 2021-05-30

analyze_date = dt.datetime(2021, 6, 1)

cltv = df[["master_id", "total_order_count"]]

cltv["recency"] = (df["last_order_date"] - df["first_order_date"]) \
    .apply(lambda date: date.days)
    
cltv["T"] = (analyze_date - df["first_order_date"]) \
    .apply(lambda date: date.days)
    
cltv["monetary"] = df["total_price"] / df["total_order_count"]

cltv.columns = ["master_id", "frequency", "recency_cltv_weekly", "T_weekly", "monetary_cltv_avg"]

cltv = cltv[cltv["frequency"] > 1]

cltv["frequency"] = cltv["frequency"].astype("int")

cltv["recency_cltv_weekly"] = cltv["recency_cltv_weekly"] / 7
cltv["T_weekly"] = cltv["T_weekly"] / 7

bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv["frequency"],
        cltv["recency_cltv_weekly"],
        cltv["T_weekly"])

cltv["exp_sales_3_month"] = bgf \
    .conditional_expected_number_of_purchases_up_to_time(12, cltv["frequency"], 
                                                         cltv["recency_cltv_weekly"], 
                                                         cltv["T_weekly"])
    
cltv["exp_sales_6_month"] = bgf \
    .conditional_expected_number_of_purchases_up_to_time(24, cltv["frequency"], 
                                                         cltv["recency_cltv_weekly"], 
                                                         cltv["T_weekly"])
plot_period_transactions(bgf)

ggf = GammaGammaFitter(penalizer_coef=0.01)

ggf.fit(cltv["frequency"], cltv["monetary_cltv_avg"])

cltv["exp_average_value"] = ggf \
    .conditional_expected_average_profit(cltv["frequency"], cltv["monetary_cltv_avg"])
 
exp_6_month_end = ggf.customer_lifetime_value(bgf, 
                                          cltv['frequency'],
                                          cltv['recency_cltv_weekly'], 
                                          cltv['T_weekly'], 
                                          cltv['monetary_cltv_avg'],
                                          time=6, # Month
                                          freq="W",
                                          discount_rate=0.01)
cltv["cltv"] = exp_6_month_end

cltv["segment"] = pd.qcut(cltv["cltv"], 4, labels=["D", "C", "B", "A"])

analyze = cltv.groupby("segment").agg({"count", "mean", "sum"})
