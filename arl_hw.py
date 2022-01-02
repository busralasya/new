

# Ödev : sepet aşamasındaki kullanıcılara ürün önerisinde bulunmak

# Kullanacağımız veri seti : Online Retail II veri seti

# İstenen : Karar kurallarını 2010-2011 Germany müşterileri üzerinden yapacağız.

# İstenen : id1 : 21987 , id2 : 23235 , id3 : 22747



import pandas as pd
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
pd.set_option("display.expand_frame_repr", False)
from mlxtend.frequent_patterns import apriori, association_rules



# Veriyi Tanımlama :

data = pd.read_excel("week-3/online_retail_II.xlsx", sheet_name="Year 2010-2011")
df = data.copy()



########################################################################################################################



# Görev 1: Veri ön işleme


def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def retail_data_prep(dataframe):
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[~dataframe["StockCode"].str.contains("POST", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]
    replace_with_thresholds(dataframe, "Quantity")
    replace_with_thresholds(dataframe, "Price")
    return dataframe

df = retail_data_prep(df)



#####################################################################################

# Görev 2 : Germany müşterileri üzerinden birliktelik kuralları üretiniz.

# BURADA SADECE GERMANY İLE İLGİLENECEĞİZ.

df = df[df["Country"] == "Germany"]

# Verinin gelmesini istediğimiz durum:

# Description   NINE DRAWER OFFICE TIDY   SET 2 TEA TOWELS I LOVE LONDON    SPACEBOY BABY GIFT SET
# Invoice
# 536370                              0                                 1                       0
# 536852                              1                                 0                       1
# 536974                              0                                 0                       0
# 537065                              1                                 0                       0
# 537463                              0                                 0                       1

# Bunu da yapmak için aşağıdaki scripti yazıyoruz.


def create_invoice_product_format( df, id = False):
    if id :
        return df.groupby(["Invoice","StockCode"])["Quantity"].sum().unstack().fillna(0).applymap( lambda x: 1 if x > 0 else 0)
    else:
        return df.groupby(["Invoice","Description"])["Quantity"].sum().unstack().fillna(0).applymap( lambda x: 1 if x > 0 else 0)

# fonksiyonu df üzerinde uyguluyoruz ve istediğimiz invoice-product formatında kullanabiliyoruz.


ge_inv_pro_df = create_invoice_product_format(df, id = True)


# Eğer stockcode ile ürünü gösterirsek bu kod ile ürünün ismine ulaşabiliriz.

def check_id(dataframe, stock_code):
    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
    print(product_name)




######################################################################

# Görev 3 : ID'leri verilen ürünlerin isimleri nelerdir?

print(check_id(df, 21987))
#['PACK OF 6 SKULL PAPER CUPS']
print(check_id(df, 23235))
#['STORAGE TIN VINTAGE LEAF']
print(check_id(df, 22747))
#["POPPY'S PLAYHOUSE BATHROOM"]



#####################################################################################

# Görev 4 : Sepetteki kullanıcılar için ürün önerisi yapınız.


#Birliktelik kuralı

frequent_itemsets = apriori(ge_inv_pro_df, min_support=0.01, use_colnames=True)
frequent_itemsets.sort_values("support", ascending=False).head(50)

rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
rules.sort_values("support", ascending=False).head()

rules.sort_values("lift", ascending=False).head(500)


#Ürün önerme


def arl_recommender(rules_df, product_id, rec_count=1):

    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []

    for i, product in sorted_rules["antecedents"].items():
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"]))
    recommendation_list = list({item for item_list in recommendation_list for item in item_list})

    return recommendation_list[:rec_count]

check_id(df,21987)
print(arl_recommender(rules, 21987, 3))
print("*******************************************")
check_id(df,23235)
print(arl_recommender(rules, 23235, 3))
print("*******************************************")
check_id(df,22747)
print(arl_recommender(rules, 22747, 3))
print("*******************************************")






#####################################################################################

# Görev 5 : Önerilen ürünlerin isimleri nelerdir?


# ['PACK OF 6 SKULL PAPER CUPS'] ' önerilen ürünler -------> ['BLUE POLKADOT PLATE '] ,['SET OF 60 PANTRY DESIGN CAKE CASES '] ve ['SPACEBOY BIRTHDAY CARD']

# ['STORAGE TIN VINTAGE LEAF'] '   önerilen ürünler -------> ['RED RETROSPOT MINI CASES'] , ['REGENCY CAKESTAND 3 TIER'] ve ['PLASTERS IN TIN SPACEBOY']

# ["POPPY'S PLAYHOUSE BATHROOM"] '   önerilen ürünler -------> ['JUMBO SHOPPER VINTAGE RED PAISLEY'] , ['PLASTERS IN TIN SPACEBOY'] ve ['PLASTERS IN TIN WOODLAND ANIMALS']

