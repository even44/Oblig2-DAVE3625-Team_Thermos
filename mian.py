import pandas as pd
import matplotlib.pyplot as plt
import datetime
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from datetime import datetime
from sklearn import metrics

# Hente data direkte fra yahooo
url = "https://query1.finance.yahoo.com/v7/finance/download/TSLA?period1=1666884371&period2=1698420371&interval=1d&events=history&includeAdjustedClose=true"
# Lese dataene med panda
df = pd.read_csv(url)


# Gjøre om datoene til faktiske datetime objekter med panda
df["Date"] = pd.to_datetime(df["Date"])
df.set_index("Date", inplace=True)

# Lage en kolonne med Ordinal dates så datoene kan representeres med heltall
df["DateOrdinal"] = df.index.map(datetime.toordinal)

# Sette DateOrdinal som input og Close som output
X = df[["DateOrdinal"]]
y = df["Close"]

# Splitter dataene inn i test og treningsdata
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Lager en modell ut av dataene
model = LinearRegression()
model.fit(X_train, y_train)

# Teste nøyaktigheten på testsettet
# og treningssettet
predictions_test = model.predict(X_test)
predictions_train = model.predict(X_train)

# Printe ut MSE
print("MSE = " + str(metrics.mean_squared_error(y_test, predictions_test)))
print("MSE = " + str(metrics.mean_squared_error(y_train, predictions_train)))


# Her kan man legge inn dato modellen skal predicte (veldig dårlig)
testdate = "2023-08-07"

specific_date = pd.to_datetime(testdate).toordinal()
predicted_price = model.predict([[specific_date]])
print(f"Predicted Price on {testdate}: {predicted_price[0]}")

# Vise lagring av modell til senere bruk
filename = "Tesla_Stock_Predictor.verynice"
pickle.dump(model, open(filename, "wb"))

# "Senere bruk"
testdate = "2024-03-01"
loaded_model = pickle.load(open(filename, "rb"))
print(loaded_model.predict([[pd.to_datetime(testdate).toordinal()]])[0])
