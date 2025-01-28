import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import backtrader as bt


file_name = "market_data2.xlsx"  # Nazwa pliku Excela


def load_excel_data(file_name):
    excel_data = pd.ExcelFile(file_name)
    data_frames = {sheet: excel_data.parse(sheet) for sheet in excel_data.sheet_names}
    print("Dostępne arkusze:", excel_data.sheet_names)
    return data_frames

data_frames = load_excel_data(file_name)


def prepare_backtest_data(df):
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')  # Konwersja daty
    df.set_index('datetime', inplace=True)
    df = df[['Open', 'High', 'Low', 'Close']].dropna()  # Wybór wymaganych kolumn
    return df

prepared_data = {sheet: prepare_backtest_data(df) for sheet, df in data_frames.items()}




import pandas as pd
import backtrader as bt


class RSI_SMAStrategy(bt.Strategy):
    params = (
        ("rsi_period", 14),
        ("rsi_overbought", 65),
        ("rsi_oversold", 30),
        ("sma_period", 40),
    )

    def __init__(self):
        self.rsi = bt.indicators.RSI(self.data.close, period=self.params.rsi_period)
        self.sma = bt.indicators.SimpleMovingAverage(self.data.close, period=self.params.sma_period)

    def next(self):
        # Kupno, gdy RSI jest poniżej progu wyprzedania i cena powyżej SMA
        if self.rsi < self.params.rsi_oversold and self.data.close > self.sma:
            self.buy()
        # Sprzedaż, gdy RSI przekracza próg wykupienia
        elif self.rsi > self.params.rsi_overbought:
            self.sell()



def run_backtest(datafile, initial_cash=10000):
    
    if datafile.endswith('.xlsx'):
        data = pd.read_excel(datafile, parse_dates=True)
    else:
        raise ValueError("Obsługiwane są tylko pliki Excel (.xlsx).")

    
    if 'Datetime' in data.columns:
        data['Datetime'] = pd.to_datetime(data['Datetime'])
        data.set_index('Datetime', inplace=True)
    else:
        raise ValueError("Plik musi zawierać kolumnę 'Datetime'.")

    
    required_columns = {'Open', 'High', 'Low', 'Close', 'Volume'}
    if not required_columns.issubset(data.columns):
        raise ValueError(f"Plik musi zawierać kolumny: {required_columns}")

    
    cerebro = bt.Cerebro()
    cerebro.broker.set_cash(initial_cash)

    
    cerebro.broker.setcommission(commission=0.001)  # 0.1% prowizji
    cerebro.broker.set_slippage_perc(0.002)         # 0.2% poślizgu

    
    data_feed = bt.feeds.PandasData(dataname=data)
    cerebro.adddata(data_feed)

    
    cerebro.addstrategy(RSI_SMAStrategy)

    
    print(f"[INFO] Kapitał początkowy: {cerebro.broker.getvalue():.2f}")
    result = cerebro.run()
    print(f"[INFO] Końcowa wartość portfela: {cerebro.broker.getvalue():.2f}")

    
    cerebro.plot()



if __name__ == "__main__":
    try:
        # Podaj nazwę pliku Excel
        datafile = "market_data2.xlsx"
        # Uruchom backtest
        run_backtest(datafile)
    except Exception as e:
        print(f"[ERROR] {e}")
