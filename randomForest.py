import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Cargar los datos
data = pd.read_csv('steam.csv', index_col='appid')

# Variables objetivo y características
y = data['positive_ratings']
features = ['release_date', 'english', 'developer', 'publisher', 'platforms', 'required_age', 'categories', 'genres', 'steamspy_tags', 'average_playtime', 'median_playtime', 'owners', 'price']
X = data[features]

# Tratamiento de valores nulos
X = X.fillna({
    'publisher': 'unknown',
    'platforms': 'unknown'
})

# Conversión de tipos de datos
X['release_date'] = pd.to_datetime(X['release_date'], errors='coerce')
X['release_year'] = X['release_date'].dt.year.fillna(0).astype(int)
X['release_month'] = X['release_date'].dt.month.fillna(0).astype(int)
X['release_day'] = X['release_date'].dt.day.fillna(0).astype(int)
X = X.drop('release_date', axis=1)

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

s = (train_X.dtypes == 'object')
object_cols = list(s[s].index)


label_X_train = train_X.copy()
label_X_valid = val_X.copy()

# Codificación de variables categóricas
ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
label_X_train[object_cols] = ordinal_encoder.fit_transform(train_X[object_cols])
label_X_valid[object_cols] = ordinal_encoder.transform(val_X[object_cols])

# Modelo de regresión con DecisionTreeRegressor
model = RandomForestRegressor(random_state=1)
model.fit(label_X_train, train_y)

# Predicciones y cálculo del error
val_predictions = model.predict(label_X_valid)
mae = mean_absolute_error(val_y, val_predictions)

print(f"Mean Absolute Error: {mae}")