import requests
import numpy as np
from scipy.signal import argrelextrema


class BinanceDataFetcher:
    def __init__(self, symbol, interval='1m'):
        """
        Инициализация класса для получения данных с Binance.

        :param symbol: Тикер пары (например, 'BTCUSDT')
        :param interval: Интервал для свечей (по умолчанию '1m')
        """
        self.symbol = symbol
        self.interval = interval
        self.base_url = "https://api.binance.com/api/v3/klines"

    def fetch_closing_prices(self, limit=100):
        """
        Получить исторические данные цен закрытия для указанной пары.

        :param limit: Количество данных для получения (максимум 1000)
        :return: Список цен закрытия
        """
        params = {
            'symbol': self.symbol,
            'interval': self.interval,
            'limit': limit
        }
        response = requests.get(self.base_url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            closing_prices = [float(item[4]) for item in data]  # Цены закрытия находятся в 4-й позиции массива
            return closing_prices
        else:
            raise Exception(f"Ошибка при запросе данных: {response.status_code} - {response.text}")

    def fetch_volume_profile(self, limit=100):
        """
        Вычислить профиль объема (простая версия на основе данных свечей).
        
        :param limit: Количество данных для получения (по умолчанию 100)
        :return: Словарь с информацией о ценах и объемах для анализа.
        """
        params = {
            'symbol': self.symbol,
            'interval': self.interval,
            'limit': limit
        }
        response = requests.get(self.base_url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            
            # Пример обработки: расчет среднего объема и цен
            volumes = [float(item[5]) for item in data]  # Объем находится в 5-й позиции массива
            prices = [float(item[4]) for item in data]   # Цена закрытия в 4-й позиции
            
            value_area = sum(volumes) / len(volumes)  # Пример расчета Value Area
            VAH = max(prices)  # Верхняя граница
            VAL = min(prices)  # Нижняя граница

            return {
                'price': prices[-1],  # Последняя цена
                'volume': sum(volumes),
                'VAH': VAH,
                'VAL': VAL,
                'ValueArea': value_area
            }
        else:
            raise Exception(f"Ошибка при запросе данных: {response.status_code} - {response.text}")

    def fetch_price_data(self, limit=100):
        """
        Получить исторические данные (high, low, close) для указанной пары.
        
        :param limit: Количество данных для получения (по умолчанию 100)
        :return: Словарь с данными о ценах: 'close', 'high', 'low'
        """
        params = {
            'symbol': self.symbol,
            'interval': self.interval,
            'limit': limit
        }
        response = requests.get(self.base_url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            closing_prices = [float(item[4]) for item in data]  # Цена закрытия
            high_prices = [float(item[2]) for item in data]    # Высокие цены
            low_prices = [float(item[3]) for item in data]     # Низкие цены

            return {
                'close': closing_prices,
                'high': high_prices,
                'low': low_prices
            }
        else:
            raise Exception(f"Ошибка при запросе данных: {response.status_code} - {response.text}")

    def fetch_ichimoku_data(self, limit=100):
        """
        Получить данные для расчета индикатора облака Ишимоку.
        
        :param limit: Количество данных для получения (по умолчанию 100)
        :return: Словарь с данными о ценах для расчета Ichimoku Cloud
        """
        data = self.fetch_price_data(limit)

        high_prices = np.array(data['high'])
        low_prices = np.array(data['low'])
        
        # Индикатор Ichimoku
        period_tenkan = 9
        period_kijun = 26
        period_senkou_span_b = 52

        # Тенкан-Сен (линия конверсии)
        tenkan_sen = (high_prices[-period_tenkan:].max() + low_prices[-period_tenkan:].min()) / 2
        
        # Киджун-Сен (базовая линия)
        kijun_sen = (high_prices[-period_kijun:].max() + low_prices[-period_kijun:].min()) / 2
        
        # Сенкоу Спан А
        senkou_span_a = (tenkan_sen + kijun_sen) / 2
        
        # Сенкоу Спан Б
        senkou_span_b = (high_prices[-period_senkou_span_b:].max() + low_prices[-period_senkou_span_b:].min()) / 2

        return {
            'tenkan_sen': tenkan_sen,
            'kijun_sen': kijun_sen,
            'senkou_span_a': senkou_span_a,
            'senkou_span_b': senkou_span_b
        }


class MACDIndicator:
    def __init__(self, short_window=12, long_window=26, signal_window=9):
        """
        Инициализация MACD индикатора.

        :param short_window: Период для короткой скользящей средней (обычно 12)
        :param long_window: Период для длинной скользящей средней (обычно 26)
        :param signal_window: Период для сигнальной линии (обычно 9)
        """
        self.short_window = short_window
        self.long_window = long_window
        self.signal_window = signal_window

    def calculate_macd(self, prices):
        """
        Рассчитать MACD, сигнальную линию и гистограмму.

        :param prices: Цены закрытия (список или numpy массив)
        :return: Кортеж (MACD линия, сигнальная линия, гистограмма)
        """
        short_ema = self._ema(prices, self.short_window)
        long_ema = self._ema(prices, self.long_window)

        macd_line = short_ema - long_ema
        signal_line = self._ema(macd_line, self.signal_window)
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    def _ema(self, prices, window):
        """
        Вычислить экспоненциальную скользящую среднюю (EMA).

        :param prices: Цены (список или numpy массив)
        :param window: Период EMA
        :return: Массив EMA значений
        """
        prices = np.array(prices, dtype=np.float64)
        ema = np.zeros_like(prices)
        multiplier = 2 / (window + 1)

        ema[0] = prices[0]  # Инициализация первым значением цен
        for i in range(1, len(prices)):
            ema[i] = (prices[i] - ema[i - 1]) * multiplier + ema[i - 1]

        return ema

    def should_open_long(self, macd_line, signal_line):
        """
        Проверить условие для открытия long сделки.

        :param macd_line: Значения MACD линии
        :param signal_line: Значения сигнальной линии
        :return: True, если условие выполнено
        """
        return (
            macd_line[-2] < 0 and macd_line[-1] < 0 and
            macd_line[-2] < signal_line[-2] and macd_line[-1] > signal_line[-1]
        )

    def should_open_short(self, macd_line, signal_line):
        """
        Проверить условие для открытия short сделки.

        :param macd_line: Значения MACD линии
        :param signal_line: Значения сигнальной линии
        :return: True, если условие выполнено
        """
        return (
            macd_line[-2] > 0 and macd_line[-1] > 0 and
            macd_line[-2] > signal_line[-2] and macd_line[-1] < signal_line[-1]
        )
    
    def main(self):
        fetcher = BinanceDataFetcher(symbol='BTCUSDT', interval='1m')
        prices = fetcher.fetch_closing_prices(limit=10)

        macd_line, signal_line, histogram = self.calculate_macd(prices)

        if self.should_open_long(macd_line, signal_line):
            print("Условие для открытия Long сделки выполнено.")
        elif self.should_open_short(macd_line, signal_line):
            print("Условие для открытия Short сделки выполнено.")
        else:
            print("Условия для открытия сделки не выполнены.")


class RSIIndicator:
    def __init__(self, period=14):
        """
        Инициализация RSI индикатора.

        :param period: Период RSI (обычно 14)
        """
        self.period = period

    def calculate_rsi(self, prices):
        """
        Рассчитать значения RSI на основе цен закрытия.

        :param prices: Цены закрытия (список или numpy массив)
        :return: Массив значений RSI
        """
        prices = np.array(prices, dtype=np.float64)
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.zeros_like(prices)
        avg_loss = np.zeros_like(prices)

        # Начальные средние значения для первого периода
        avg_gain[self.period] = np.mean(gains[:self.period])
        avg_loss[self.period] = np.mean(losses[:self.period])

        # Итерация по всем остальным ценам
        for i in range(self.period + 1, len(prices)):
            avg_gain[i] = (avg_gain[i - 1] * (self.period - 1) + gains[i - 1]) / self.period
            avg_loss[i] = (avg_loss[i - 1] * (self.period - 1) + losses[i - 1]) / self.period

        # Обработка деления на ноль
        rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
        rsi = 100 - (100 / (1 + rs))

        rsi[:self.period] = np.nan  # Первые значения RSI недоступны
        return rsi

    def is_overbought(self, rsi):
        """
        Проверить перекупленность.

        :param rsi: Массив значений RSI
        :return: True, если последний RSI > 70
        """
        return rsi[-1] > 70

    def is_oversold(self, rsi):
        """
        Проверить перепроданность.

        :param rsi: Массив значений RSI
        :return: True, если последний RSI < 30
        """
        return rsi[-1] < 30

    def detect_divergence(self, prices, rsi):
        """
        Проверить наличие дивергенции между ценами и RSI.

        :param prices: Цены закрытия (список или numpy массив)
        :param rsi: Массив значений RSI
        :return: Строка, описывающая тип дивергенции (если есть)
        """
        prices = np.array(prices)
        rsi = np.array(rsi)

        # Локальные максимумы и минимумы
        price_highs = argrelextrema(prices, np.greater)[0]
        price_lows = argrelextrema(prices, np.less)[0]
        rsi_highs = argrelextrema(rsi, np.greater)[0]
        rsi_lows = argrelextrema(rsi, np.less)[0]

        # Проверка дивергенции
        for i in price_highs:
            if i in rsi_highs and prices[i] > prices[i - 1] and rsi[i] < rsi[i - 1]:
                return "Медвежья дивергенция: цена растет, а RSI падает."

        for i in price_lows:
            if i in rsi_lows and prices[i] < prices[i - 1] and rsi[i] > rsi[i - 1]:
                return "Бычья дивергенция: цена падает, а RSI растет."

        return "Дивергенция не обнаружена."

    def main(self):
        fetcher = BinanceDataFetcher(symbol='BTCUSDT', interval='4h')
        prices = fetcher.fetch_closing_prices(limit=50)

        if len(prices) < self.period:
            raise ValueError("Недостаточно данных для расчета RSI.")

        rsi = self.calculate_rsi(prices)

        # Проверка перепроданности и перекупленности
        if self.is_oversold(rsi):
            print("Актив перепродан, следует покупать.")
        elif self.is_overbought(rsi):
            print("Актив перекуплен, следует продавать.")
        else:
            # Если не перепродан и не перекуплен, проверяем на дивергенцию
            divergence = self.detect_divergence(prices, rsi)
            print(divergence)


class MovingAverageStrategy:
    def __init__(self, ma_short=20, ma_long=50):
        self.ma_short = ma_short
        self.ma_long = ma_long

    def calculate_moving_average(self, prices, period):
        """
        Вычислить скользящую среднюю.

        :param prices: Цены закрытия (список или numpy массив)
        :param period: Период для расчета MA
        :return: Массив значений MA
        """
        return np.convolve(prices, np.ones(period) / period, mode='valid')

    def check_buy_condition(self, prices, ma_short, ma_long):
        """
        Проверить условия для покупки.

        :param prices: Цены закрытия
        :param ma_short: Краткосрочная MA
        :param ma_long: Долгосрочная MA
        :return: True, если условия выполнены
        """
        if prices[-1] > ma_short[-1] > ma_long[-1]:
            if ma_short[-1] > ma_long[-1]:
                if (prices[-2] <= ma_short[-2] or prices[-2] <= ma_long[-2]) and prices[-1] > ma_short[-1]:
                    return True
        return False

    def check_sell_condition(self, prices, ma_short, ma_long):
        """
        Проверить условия для продажи.

        :param prices: Цены закрытия
        :param ma_short: Краткосрочная MA
        :param ma_long: Долгосрочная MA
        :return: True, если условия выполнены
        """
        if prices[-1] < ma_short[-1] < ma_long[-1]:
            if ma_short[-1] < ma_long[-1]:
                if (prices[-2] >= ma_short[-2] or prices[-2] >= ma_long[-2]) and prices[-1] < ma_short[-1]:
                    return True
        return False

    def main(self):
        fetcher = BinanceDataFetcher(symbol='BTCUSDT', interval='4h')
        prices = fetcher.fetch_closing_prices(limit=100)

        if len(prices) < max(self.ma_short, self.ma_long):
            raise ValueError("Недостаточно данных для расчета скользящих средних.")

        ma_short = self.calculate_moving_average(prices, self.ma_short)
        ma_long = self.calculate_moving_average(prices, self.ma_long)

        # Скользящие средние могут быть короче исходного массива цен
        prices = prices[len(prices) - len(ma_short):]

        if self.check_buy_condition(prices, ma_short, ma_long):
            print("Условия для покупки выполнены.")
        elif self.check_sell_condition(prices, ma_short, ma_long):
            print("Условия для продажи выполнены.")
        else:
            print("Условия для сделки не выполнены.")


class FibonacciStrategy:
    def __init__(self):
        self.levels = [0.0, 0.382, 0.5, 0.618, 0.786, 1.0]

    def find_extrema(self, prices):
        """
        Найти минимумы и максимумы в ценах.

        :param prices: Массив цен закрытия
        :return: Минимум и максимум
        """
        local_min = np.min(prices)
        local_max = np.max(prices)
        return local_min, local_max

    def calculate_fibonacci_levels(self, min_price, max_price):
        """
        Рассчитать уровни Фибоначчи.

        :param min_price: Минимальная цена (0%)
        :param max_price: Максимальная цена (100%)
        :return: Словарь уровней Фибоначчи
        """
        return {
            level: min_price + (max_price - min_price) * level
            for level in self.levels
        }

    def check_trade_conditions(self, prices, fib_levels):
        """
        Проверить условия входа в сделку.

        :param prices: Массив цен закрытия
        :param fib_levels: Уровни Фибоначчи
        :return: Условие для покупки или продажи
        """
        last_price = prices[-1]
        for level, price in fib_levels.items():
            if abs(last_price - price) < 0.001 * last_price:  # Допуск в 0.1%
                return True
        return False

    def main(self):
        fetcher = BinanceDataFetcher(symbol='BTCUSDT', interval='4h')
        prices = fetcher.fetch_closing_prices(limit=100)

        if len(prices) < 2:
            raise ValueError("Недостаточно данных для анализа.")

        local_min, local_max = self.find_extrema(prices)
        fib_levels = self.calculate_fibonacci_levels(local_min, local_max)

        if self.check_trade_conditions(prices, fib_levels):
            print("Условия для сделки на уровне Фибоначчи выполнены.")
        else:
            print("Условия для сделки на уровне Фибоначчи не выполнены.")


class VolumeProfileTrader:
    def __init__(self, symbol='BTCUSDT', interval='1h'):
        """
        Инициализация трейдера для работы с Volume Profile.

        :param symbol: Тикер пары для Binance (по умолчанию 'BTCUSDT')
        :param interval: Интервал для данных (по умолчанию '1h')
        """
        self.symbol = symbol
        self.interval = interval
        self.binance_fetcher = BinanceDataFetcher(symbol=self.symbol, interval=self.interval)

    def get_volume_profile_data(self):
        """
        Получаем данные профиля объема.
        """
        return self.binance_fetcher.fetch_volume_profile()

    def analyze_and_trade(self):
        """
        Анализирует данные профиля объема и принимает решение о торговле.
        """
        # Получаем данные Volume Profile
        profile_data = self.get_volume_profile_data()

        # Гипотеза для покупки
        if self.check_buy_condition(profile_data):
            return "Buy"
        
        # Гипотеза для продажи
        if self.check_sell_condition(profile_data):
            return "Sell"

        return "Hold"

    def check_buy_condition(self, profile_data):
        """
        Проверка гипотезы для покупки:
        Цена выходит в зону VAH и возвращается в зону стоимости (Value Area).
        """
        if profile_data['price'] > profile_data['VAH'] and profile_data['price'] < profile_data['ValueArea']:
            return True
        return False

    def check_sell_condition(self, profile_data):
        """
        Проверка гипотезы для продажи:
        Цена выходит в зону VAL и возвращается в зону стоимости (Value Area).
        """
        if profile_data['price'] < profile_data['VAL'] and profile_data['price'] > profile_data['ValueArea']:
            return True
        return False

    def main(self):
        """
        Главная функция для анализа и принятия торгового решения.
        """
        action = self.analyze_and_trade()
        print(f"Торговое решение: {action}")


class IchimokuCloud:
    def __init__(self, symbol='BTCUSDT', interval='1h'):
        """
        Инициализация индикатора облака Ишимоку.

        :param symbol: Тикер пары (например, 'BTCUSDT')
        :param interval: Интервал для свечей (например, '1h', '5m', '1d')
        """
        self.symbol = symbol
        self.interval = interval
        self.base_url = "https://api.binance.com/api/v3/klines"

    def fetch_data(self, limit=100):
        """
        Получить исторические данные свечей.

        :param limit: Количество данных для получения (например, 100)
        :return: Данные свечей
        """
        params = {
            'symbol': self.symbol,
            'interval': self.interval,
            'limit': limit
        }
        response = requests.get(self.base_url, params=params)
        if response.status_code == 200:
            data = response.json()
            return np.array([[float(item[4]) for item in data]])  # Цены закрытия
        else:
            raise Exception(f"Ошибка при запросе данных: {response.status_code} - {response.text}")

    def calculate_ichimoku(self, prices):
        """
        Рассчитать индикаторы облака Ишимоку: Тенкан-Сен, Киджун-Сен, Сенко-Спан-А и Сенко-Спан-Б.

        :param prices: Цены закрытия
        :return: Линии Ишимоку
        """
        tenkan_sen = self.calculate_tenkan_sen(prices)
        kijun_sen = self.calculate_kijun_sen(prices)
        senkou_span_a, senkou_span_b = self.calculate_senkou_span(prices)

        return tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b

    def calculate_tenkan_sen(self, prices):
        """
        Рассчитать Тенкан-Сен (линия переворота).
        :param prices: Цены
        :return: Линия Тенкан-Сен
        """
        return (np.max(prices[-9:]) + np.min(prices[-9:])) / 2

    def calculate_kijun_sen(self, prices):
        """
        Рассчитать Киджун-Сен (базовая линия).
        :param prices: Цены
        :return: Линия Киджун-Сен
        """
        return (np.max(prices[-26:]) + np.min(prices[-26:])) / 2

    def calculate_senkou_span(self, prices):
        """
        Рассчитать линии Сенко-Спан-А и Сенко-Спан-Б.
        :param prices: Цены
        :return: Линии Сенко-Спан-А и Сенко-Спан-Б
        """
        senkou_span_a = (self.calculate_tenkan_sen(prices) + self.calculate_kijun_sen(prices)) / 2
        senkou_span_b = (np.max(prices[-52:]) + np.min(prices[-52:])) / 2
        return senkou_span_a, senkou_span_b

    def analyze_and_trade(self):
        """
        Анализирует рынок и принимает решение на основе облака Ишимоку.
        """
        prices = self.fetch_data(limit=100)
        tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b = self.calculate_ichimoku(prices)

        last_price = prices[-1]  # Получаем последнюю цену

        # Получаем последние значения из облака Ишимоку
        last_senkou_span_a = senkou_span_a[-1]
        last_senkou_span_b = senkou_span_b[-1]

        # Сравниваем последнюю цену с линиями Ишимоку
        if last_price > last_senkou_span_a and last_price < last_senkou_span_b:
            print("Цена внутри облака — воздержитесь от сделок.")
        elif tenkan_sen[-1] > kijun_sen[-1]:
            print("Открыть длинную позицию.")
        elif tenkan_sen[-1] < kijun_sen[-1]:
            print("Открыть короткую позицию.")
        else:
            print("Нет явного сигнала.")

        
    def main(self):
        """
        Основной метод, который может быть вызван для анализа и совершения сделки.
        """
        self.analyze_and_trade()


class IchimokuStrategy:
    def __init__(self, symbol='BTCUSDT', interval='1m', limit=100):
        self.symbol = symbol
        self.interval = interval
        self.limit = limit
        self.data_fetcher = BinanceDataFetcher(symbol, interval)

    def get_ichimoku_data(self):
        """
        Получить данные для расчета индикатора Ichimoku Cloud.
        """
        ichimoku_data = self.data_fetcher.fetch_ichimoku_data(self.limit)
        return ichimoku_data

    def get_last_price(self):
        """
        Получить последнюю цену закрытия.
        """
        prices = self.data_fetcher.fetch_closing_prices(self.limit)
        return prices[-1]

    def analyze_and_trade(self):
        """
        Анализирует рынок и принимает решение о торговле.
        """
        ichimoku_data = self.get_ichimoku_data()
        last_price = self.get_last_price()

        tenkan_sen = ichimoku_data['tenkan_sen']
        kijun_sen = ichimoku_data['kijun_sen']
        senkou_span_a = ichimoku_data['senkou_span_a']
        senkou_span_b = ichimoku_data['senkou_span_b']

        # Позиция цены относительно облака Ишимоку
        if senkou_span_a > senkou_span_b:  # Цена выше облака
            cloud_position = 'bullish'
        else:  # Цена ниже облака
            cloud_position = 'bearish'

        # Стратегия на основе пересечения Тенкан-Сен и Киджун-Сен
        if tenkan_sen > kijun_sen:
            crossover_signal = 'bullish'
        else:
            crossover_signal = 'bearish'

        # Проверка, находится ли цена внутри облака Ишимоку
        if senkou_span_a < last_price < senkou_span_b:
            print("Цена находится внутри облака Ишимоку - воздержитесь от сделок")
            return "Hold"

        # Логика для открытия длинной позиции
        if cloud_position == 'bullish' and crossover_signal == 'bullish':
            print("Открыть длинную позицию: Цена выше облака и Тенкан-Сен пересекает Киджун-Сен снизу вверх")
            # Здесь можно добавить логику для открытия ордера на покупку
            return "Buy"

        # Логика для открытия короткой позиции
        elif cloud_position == 'bearish' and crossover_signal == 'bearish':
            print("Открыть короткую позицию: Цена ниже облака и Тенкан-Сен пересекает Киджун-Сен сверху вниз")
            # Здесь можно добавить логику для открытия ордера на продажу
            return "Sell"

        # Логика для закрытия позиции
        elif crossover_signal == 'bearish' and cloud_position == 'bearish':
            print("Закрыть позицию: Тенкан-Сен пересекает Киджун-Сен сверху вниз - завершение тренда")
            # Здесь можно добавить логику для закрытия позиции
            return "Close"

        print("Нет сигнала для открытия или закрытия позиции")
        return "Hold"

    def main(self):
        """
        Основная функция для выполнения торговой логики.
        """
        signal = self.analyze_and_trade()
        print(f"Рекомендация: {signal}")


# Пример использования
if __name__ == "__main__":
    # Пример данных цен закрытия
    macd_indicator = IchimokuStrategy()
    macd_indicator.main()