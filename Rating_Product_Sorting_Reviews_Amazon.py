

import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

df = pd.read_csv("amazon_review.csv")

df.head()
df.shape

df["overall"].mean()



df.describe().T

(df.loc[df["day_diff"] <= 30, "overall"].mean() * 29/100 + \
    df.loc[(df["day_diff"] > 30) & (df["day_diff"] <= 90), "overall"].mean() * 27/100 + \
    df.loc[(df["day_diff"] > 90) & (df["day_diff"] <= 180), "overall"].mean() * 25/100 + \
    df.loc[df["day_diff"] > 180, "overall"].mean() * 19/100)

#alternatif
birinci_aralik = df["day_diff"].quantile(0.25)
ikinci_aralik = df["day_diff"].quantile(0.50)
ucuncu_aralik = df["day_diff"].quantile(0.75)
def time_based_weighted_average(dataframe, w1 = 30, w2 = 28, w3 = 26, w4 = 24):
    return dataframe.loc[dataframe["day_diff"] <= birinci_aralik, "overall"].mean() * w1 / 100 + \
    dataframe.loc[(dataframe["day_diff"] > birinci_aralik) &(dataframe["day_diff"] <= ikinci_aralik) , "overall"].mean() * w2 / 100 + \
    dataframe.loc[(dataframe["day_diff"] > ikinci_aralik) &(dataframe["day_diff"] <= ucuncu_aralik) , "overall"].mean() * w3 / 100 + \
    dataframe.loc[dataframe["day_diff"] > ucuncu_aralik, "overall"].mean() * w4 / 100


#alternatif
df.loc[df["day_diff"] <= 100, "overall"].mean() * 22/100 + \
df.loc[(df["day_diff"] > 100) & (df["day_diff"] <= 400), "overall"].mean() * 26/100 + \
df.loc[(df["day_diff"] > 400) & (df["day_diff"] <= 700), "overall"].mean() * 28/100 + \
df.loc[(df["day_diff"] > 700) & (df["day_diff"] <= 1100), "overall"].mean() * 24/100


df.loc[df["day_diff"] <= 30, "overall"].mean()

df.loc[(df["day_diff"] > 30) & (df["day_diff"] <= 90), "overall"].mean()

df.loc[(df["day_diff"] > 90) & (df["day_diff"] <= 180), "overall"].mean()

df.loc[df["day_diff"] > 180, "overall"].mean()



df.head()


df["helpful_no"] = df["total_vote"] - df["helpful_yes"]


def score_up_down_diff(up, down):
    return up - down

# score_pos_neg_diff
df["score_pos_neg_diff"] = df.apply(lambda x: score_up_down_diff(x["helpful_yes"],
                                                                 x["helpful_no"]), axis=1)
def score_average_rating(up, down):
    if up + down == 0:
        return 0
    return up / (up + down)

# score_average_rating
df["score_average_rating"] = df.apply(lambda x: score_average_rating(x["helpful_yes"],
                                                                     x["helpful_no"]), axis=1)


def wilson_lower_bound(up, down, confidence=0.95):
    """
    Wilson Lower Bound Score hesapla

    - Bernoulli parametresi p için hesaplanacak güven aralığının alt sınırı WLB skoru olarak kabul edilir.
    - Hesaplanacak skor ürün sıralaması için kullanılır.
    - Not:
    Eğer skorlar 1-5 arasıdaysa 1-3 negatif, 4-5 pozitif olarak işaretlenir ve bernoulli'ye uygun hale getirilebilir.
    Bu beraberinde bazı problemleri de getirir. Bu sebeple bayesian average rating yapmak gerekir.

    Parameters
    ----------
    up: int
        up count
    down: int
        down count
    confidence: float
        confidence

    Returns
    -------
    wilson score: float

    """
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)

# wilson_lower_bound
df["wilson_lower_bound"] = df.apply(lambda x: wilson_lower_bound(x["helpful_yes"],
                                                                 x["helpful_no"]), axis=1)



df.sort_values("wilson_lower_bound", ascending=False).head(20)

def calculate_bayesian_rating_products(rating_counts,confidence_level=0.95):
    if sum(rating_counts)==0:
        return 0
    # Calculate the expected expected value of the rating distribution
    num_scores=len(rating_counts)
    z=st.norm.ppf(1-(1-confidence_level)/2)
    total_ratings=sum(rating_counts)
    expected_value=0.0
    expected_value_squared=0.0
    for score,count in enumerate(rating_counts):
        probability=(count+1)/(total_ratings+num_scores)
        expected_value += (score + 1) * probability
        expected_value_squared += (score + 1) * (score + 1) * probability
    # Calculate the variance of the rating distribution
    variance=(expected_value_squared-expected_value **2)/(total_ratings+num_scores+1)
    # Calculate the Bayesian avg score
    bayesian_average=expected_value-z*math.sqrt(variance)
    return bayesian_average
