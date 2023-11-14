import warnings

import helpers
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_validate
import warnings
pd.set_option("display.max_columns", 20)
pd.set_option('display.expand_frame_repr', False)
pd.set_option("display.float_format", lambda x: "%.3f" % x)
warnings.simplefilter(action="ignore", category=Warning)

def load_dataset():
    data = pd.read_csv("hitter_dataset.csv")
    return data

df = load_dataset()

# cat, num, car değişkenlere ayırıyoruz
cat_cols, num_cols, car_cols = helpers.grab_col_names(df, cat_th=7, car_th=10)

# Check ediyoruz num_colsları
helpers.check_df(df[num_cols])

# num colslar tek figurede incelendi
helpers.num_summary(df, num_cols, plot=True)


# Categorical datalar yüzdeler biçiminde gözlemlendi
for col in cat_cols:
    helpers.cat_summary(df, col, plot=True)


# outlierlar incelenmesi üzerine yazılan fonksiyonda outlier içeren veriler incelendi (yok)

for col in cat_cols:
    helpers.target_summary_with_cat(df, target="Salary", categorical_col=col)

# Target Değişkenimizin cat_colslar ilişkisi gözlemlendi.


for col in num_cols:
    helpers.target_count_with_num(df, target="League", num_cols=col)

# Target Değişkenimizin num_colslarla ilişkisi gözlemlendi



# Outliers değerimiz gözükmüyor.

for col in num_cols:
    print(col, helpers.check_outlier(dataframe=df, col_name=col, q1=0.05, q3=0.95))

for col in num_cols:
    idx = print("\n",f"-------- {col} Outliers ------------", helpers.grab_outliers(df, col, index=True), "\n")



helpers.high_correlated_cols(df[num_cols], corr_th=0.90, plot=True)

# her şey neredeyse salary'e pozitif anlamda etki etmektedir.

# CAtBat ve Chits corelasyonu 1 yani gereksiz 1 veri ifadesi biri ile açıklanabiliyor her şey


for col in [col for col in df.columns if df[col].dtypes in ["int64", "float64"]]:
    helpers.num_summary(df, col, plot=True)

for col in num_cols:
    sns.boxplot(x=df[col])
    plt.show(block=True)


# Missing Values target featureda oldugundan drop ediyoruz

df.dropna(inplace=True)

# Local Outlier Factor ile ele alıyoruz

clf = LocalOutlierFactor(n_neighbors=20)
clf.fit_predict(df[num_cols])

df_scores = clf.negative_outlier_factor_
df_scores[0:5]

scores = pd.DataFrame(np.sort(df_scores))
scores.plot(stacked=True, xlim=[0,50], style="-*")
plt.show(block=True)

# Threshold değeri grafiksel anlamda incelenip sert geçiş yapılan yerde seçildi.

th_loc = np.sort(df_scores)[5]

# outlier değerlerini verisetimizden sildik.

df.drop(labels = df[df_scores < th_loc].index, inplace=True)


# one hot encoding.

dff = helpers.one_hot_encoder(df, cat_cols)

# Robust Scaling

from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()

dff = pd.DataFrame(scaler.fit_transform(dff), columns=dff.columns)


X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=13, test_size=0.2, shuffle=True)


y = dff["Salary"]
X = dff.drop("Salary", axis=1)


from sklearn.neighbors import KNeighborsRegressor

knn_model = KNeighborsRegressor().fit(X_train,y_train)

y_pred = knn_model.predict(X_test)

mean_squared_error(y_test, y_pred)
###


# Train hatası incelensin - model ezberlenmiş
cart_model = DecisionTreeRegressor(random_state=1).fit(X_train, y_train)
y_pred_cart_train = cart_model.predict(X_train)
mean_squared_error(y_train,y_pred_cart_train)
# 0 MSE


# Test hatamıza bakıyoruz
cart_model = DecisionTreeRegressor(random_state=1).fit(X_train, y_train)
y_pred_cart_test = cart_model.predict(X_test)
mean_squared_error(y_test, y_pred_cart_test)
# 0.27 MSE


# Cross Validation Yaparız.

cart_model = DecisionTreeRegressor(random_state=1).fit(X_train, y_train)

cv_result = cross_validate(cart_model, X, y, cv=5, verbose=1, n_jobs=-1, scoring=["neg_mean_squared_error", "neg_mean_absolute_error"])

-cv_result["test_neg_mean_squared_error"].mean()
-cv_result["test_neg_mean_absolute_error"].mean()