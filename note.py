## Note
## Categorical + numerical col ları ile;
import pandas as pd
# pd.get_dummies(df[cat_cols+num_cols],drop_first=True)
# Sonra scaling işlemi minmax veya robust scaler
# Robustscaler
from sklearn.preprocessing import RobustScaler
from sklearn.impute import KNNImputer
scaler = RobustScaler()
# pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
imputer = KNNImputer(n_neighbors=5)
# df = pd.DataFrame(imputer.fit_transform(df), columns = df.columns)

### Reverse

# df = pd.DataFrame(scaler.inverse_transform(df), columns = df.columns)

# Dolu hali suan net bir şekilde gözüküyor. Daha iyi görmek için;

# df["col_changed"] = dff["col"] burada kıyas yapıyoruz.

# df.loc[df["col"].isnull(), ["col_changed", "col"]]


import missingno as msno

msno.matrix(df)
plt.show(block=True)

msno.heatmap(df)
plt.show(block=True)
