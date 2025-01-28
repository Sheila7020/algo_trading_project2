
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import backtrader as bt


file_name = "market_data2.xlsx"
excel_data = pd.ExcelFile(file_name)


data_frames = {sheet: excel_data.parse(sheet) for sheet in excel_data.sheet_names}
print("Dostępne arkusze:", excel_data.sheet_names)


def prepare_data(df):
    df = df.copy()


    if 'Datetime' in df.columns:
        df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')


    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()


    def calculate_rsi(series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    df['RSI_14'] = calculate_rsi(df['Close'])


    bb_period = 20
    df['BB_Middle'] = df['Close'].rolling(window=bb_period).mean()
    df['BB_Upper'] = df['BB_Middle'] + 2 * df['Close'].rolling(window=bb_period).std()
    df['BB_Lower'] = df['BB_Middle'] - 2 * df['Close'].rolling(window=bb_period).std()


    df = df.dropna()


    df['Signal'] = (df['Close'] > df['SMA_50']).astype(int)

    return df


prepared_data = {sheet: prepare_data(df) for sheet, df in data_frames.items()}


analysis_results = {}

for sheet, df in prepared_data.items():
    print(f"Analiza dla arkusza: {sheet}")


    X = df[['SMA_50', 'SMA_200', 'RSI_14', 'BB_Middle', 'BB_Upper', 'BB_Lower']]
    y = df['Signal']


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)


    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    analysis_results[sheet] = accuracy


    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X)
    model.fit(X_reduced, y)


    xx, yy = np.meshgrid(
        np.linspace(X_reduced[:, 0].min(), X_reduced[:, 0].max(), 100),
        np.linspace(X_reduced[:, 1].min(), X_reduced[:, 1].max(), 100)
    )
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)


    plt.contourf(xx, yy, Z, alpha=0.75, cmap=plt.cm.RdYlBu)
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, edgecolors='k', marker='o', cmap=plt.cm.RdYlBu, s=50)
    plt.title(f"Granica decyzyjna - {sheet}")
    plt.xlabel("Główna składowa 1")
    plt.ylabel("Główna składowa 2")
    plt.colorbar(label='Sygnał (0 = Sprzedaż, 1 = Kupno)')
    plt.show()


print("Dokładności modelu dla każdego arkusza:")
for sheet, acc in analysis_results.items():
    print(f"{sheet}: {acc:.2f}")


