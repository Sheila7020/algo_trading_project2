import backtrader as bt
import numpy as np
import pandas as pd
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

warnings.simplefilter(action='ignore', category=UserWarning)


class MachineLearningStrategy(bt.Strategy):
    params = dict(
        rsi_period=14,
        macd_short=12,
        macd_long=26,
        macd_signal=9,
        atr_period=14,
        capital_per_trade=0.1,
        max_positions=10,
        risk_free_rate=0.03,
        stop_loss=0.02,
        take_profit=0.05
    )

    def __init__(self):
        self.rsi = bt.indicators.RSI(period=self.params.rsi_period)
        self.macd = bt.indicators.MACD(period_me1=self.params.macd_short,
                                       period_me2=self.params.macd_long,
                                       period_signal=self.params.macd_signal)
        self.atr = bt.indicators.ATR(period=self.params.atr_period)
        self.clf = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.train_model()
        self.returns = []
        self.equity_curve = []
        self.wins = 0
        self.losses = 0
        self.initial_cash = None
        self.entry_prices = {}

    def train_model(self):
        df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
        df.rename(columns={'Max': 'High', 'Min': 'Low'}, inplace=True)
        df['rsi'] = df['Close'].rolling(self.params.rsi_period).apply(
            lambda x: (100 - (100 / (1 + (x.mean() / (x.std() + 1e-5))))))
        df['macd'] = df['Close'].ewm(span=self.params.macd_short).mean() - df['Close'].ewm(
            span=self.params.macd_long).mean()
        df['atr'] = df['High'] - df['Low']
        df.dropna(inplace=True)
        df['target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
        features = df[['rsi', 'macd', 'atr']]
        target = df['target']
        features_scaled = self.scaler.fit_transform(features)
        self.clf.fit(features_scaled, target)

    def next(self):
        if self.initial_cash is None:
            self.initial_cash = self.broker.get_cash()

        cash = self.broker.get_cash()
        price = self.data.close[0]
        size = max(1, int((cash * self.params.capital_per_trade) / price))

        if len(self.broker.positions) >= self.params.max_positions or cash < price:
            return

        features = np.array([[self.rsi[0], self.macd.macd[0], self.atr[0]]])
        features_scaled = self.scaler.transform(features)
        prediction = self.clf.predict(features_scaled)

        if not self.position:
            if prediction == 1:
                self.buy(size=size)
                self.entry_prices[self.data] = price
            else:
                self.sell(size=size)
                self.entry_prices[self.data] = price

        if self.position:
            entry_price = self.entry_prices.get(self.data, price)
            if price <= entry_price * (1 - self.params.stop_loss) or price >= entry_price * (
                    1 + self.params.take_profit):
                self.close()
                self.wins += int(price > entry_price)
                self.losses += int(price < entry_price)

        self.equity_curve.append(self.broker.get_value())
        if len(self.equity_curve) > 1:
            self.returns.append((self.equity_curve[-1] / self.equity_curve[-2]) - 1)

    def stop(self):
        returns_array = np.array(self.returns)
        sharpe_ratio = ((np.nanmean(returns_array) - (self.params.risk_free_rate / 252)) /
                        np.nanstd(returns_array)) * np.sqrt(252) if len(returns_array) > 1 else 0.0
        running_max = np.maximum.accumulate(self.equity_curve)
        max_drawdown = np.min((self.equity_curve - running_max) / running_max) if len(self.equity_curve) > 1 else 0.0
        total_return = (self.broker.get_value() - self.initial_cash) / self.initial_cash * 100
        win_rate = (self.wins / (self.wins + self.losses)) * 100 if (self.wins + self.losses) > 0 else 0.0

        print(f'Sharpe Ratio: {sharpe_ratio:.4f}')
        print(f'Max Drawdown: {max_drawdown:.4f}')
        print(f'Total Return: {total_return:.2f}%')
        print(f'Win Rate: {win_rate:.2f}%')


file_path = "path"
data = pd.read_csv(file_path, index_col='Date', parse_dates=True)
data.rename(columns={'Max': 'High', 'Min': 'Low'}, inplace=True)
bt_data = bt.feeds.PandasData(dataname=data)

cerebro = bt.Cerebro()
cerebro.addstrategy(MachineLearningStrategy)
cerebro.adddata(bt_data)
cerebro.broker.set_cash(10000)
cerebro.broker.setcommission(commission=0.004)
cerebro.broker.set_slippage_fixed(0.002)
cerebro.addsizer(bt.sizers.FixedSize, stake=1)

cerebro.run()
cerebro.plot()
