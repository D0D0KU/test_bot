import os
import sys

def find_src_path():
    # Получить абсолютный путь к текущему файлу
    current_file_path = os.path.abspath(__file__)

    # Подниматься по дереву директорий, пока не найдется директория "src"
    while True:
        current_file_path, tail = os.path.split(current_file_path)
        if tail == "src":
            return current_file_path

        # Проверка на корень файловой системы
        if not tail:
            raise FileNotFoundError("Директория 'src' не найдена в структуре проекта.")

def prepare_environment():
    src_path = find_src_path()
    sys.path.append(src_path)

prepare_environment()


import os
import sys
import json
from datetime import datetime, timedelta
from src.indicators.indicators import BinanceDataFetcher, MACDIndicator, RSIIndicator, MovingAverageStrategy, FibonacciStrategy, VolumeProfileTrader, IchimokuCloud

# Настройки
SYMBOL = "BTCUSDT"
INTERVAL = "1h"  # Интервал для анализа
DAYS = 1  # Количество дней для анализа
LIMIT_PER_INTERVAL = 100  # Количество свечей, которые запрашиваются за раз

# Генерация временного диапазона с учётом INTERVAL
def generate_time_range(days, interval):
    """
    Создает временной диапазон с учетом выбранного интервала.

    :param days: Количество дней анализа.
    :param interval: Интервал Binance (например, '1m', '1h', '1d').
    :return: Список временных меток для заданного интервала.
    """
    interval_map = {
        "1m": timedelta(minutes=1),
        "3m": timedelta(minutes=3),
        "5m": timedelta(minutes=5),
        "15m": timedelta(minutes=15),
        "30m": timedelta(minutes=30),
        "1h": timedelta(hours=1),
        "2h": timedelta(hours=2),
        "4h": timedelta(hours=4),
        "6h": timedelta(hours=6),
        "8h": timedelta(hours=8),
        "12h": timedelta(hours=12),
        "1d": timedelta(days=1),
        "3d": timedelta(days=3),
        "1w": timedelta(weeks=1),
        "1M": timedelta(days=30),  # Приблизительно месяц
    }

    if interval not in interval_map:
        raise ValueError(f"Интервал {interval} не поддерживается. Укажите правильный интервал.")

    step = interval_map[interval]
    now = datetime.now()
    start_time = now - timedelta(days=days)

    time_range = []
    current_time = start_time
    while current_time <= now:
        time_range.append(current_time)
        current_time += step

    return time_range

# Основная функция
def run_indicators():
    """
    Запускает все индикаторы для анализа данных и записывает результаты в файл.
    """
    # Инициализация индикаторов и трейдеров
    fetcher = BinanceDataFetcher(symbol=SYMBOL, interval=INTERVAL)
    macd = MACDIndicator()
    rsi = RSIIndicator()
    ma_strategy = MovingAverageStrategy()
    fib_strategy = FibonacciStrategy()
    volume_profile_trader = VolumeProfileTrader(symbol=SYMBOL, interval=INTERVAL)
    ichimoku_cloud = IchimokuCloud(symbol=SYMBOL, interval=INTERVAL)

    results = []

    try:
        # Генерация временного диапазона
        time_range = generate_time_range(DAYS, INTERVAL)

        for current_time in time_range:
            # Получаем данные свечей
            closing_prices = fetcher.fetch_closing_prices(limit=LIMIT_PER_INTERVAL)

            # Расчет индикаторов
            macd_line, signal_line, histogram = macd.calculate_macd(closing_prices)
            macd_action = "Hold"
            if macd_line[-1] < 0 and macd_line[-1] > signal_line[-1] and macd_line[-2] <= signal_line[-2]:
                macd_action = "Buy"
            elif macd_line[-1] > 0 and macd_line[-1] < signal_line[-1] and macd_line[-2] >= signal_line[-2]:
                macd_action = "Sell"

            rsi_values = rsi.calculate_rsi(closing_prices)
            rsi_action = "Hold"
            if rsi.is_oversold(rsi_values):
                rsi_action = "Buy"
            elif rsi.is_overbought(rsi_values):
                rsi_action = "Sell"
            else:
                rsi_action = rsi.detect_divergence(closing_prices, rsi_values)

            ma_short = ma_strategy.calculate_moving_average(closing_prices, ma_strategy.ma_short)
            ma_long = ma_strategy.calculate_moving_average(closing_prices, ma_strategy.ma_long)
            ma_action = "Hold"
            if ma_strategy.check_buy_condition(closing_prices, ma_short, ma_long):
                ma_action = "Buy"
            elif ma_strategy.check_sell_condition(closing_prices, ma_short, ma_long):
                ma_action = "Sell"

            local_min, local_max = fib_strategy.find_extrema(closing_prices)
            fib_levels = fib_strategy.calculate_fibonacci_levels(local_min, local_max)
            fib_action = "Hold"
            if fib_strategy.check_trade_conditions(closing_prices, fib_levels):
                fib_action = "Action on Fibonacci Level"

            profile_data = volume_profile_trader.get_volume_profile_data()
            profile_price = profile_data.get('price', None)
            VAH = profile_data.get('VAH', None)
            VAL = profile_data.get('VAL', None)
            value_area = profile_data.get('ValueArea', None)
            volume_profile_action = "Hold"
            if profile_price and VAH and VAL:
                if profile_price > VAH:
                    volume_profile_action = "Sell"
                elif profile_price < VAL:
                    volume_profile_action = "Buy"

            tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b = ichimoku_cloud.calculate_ichimoku(closing_prices)
            ichimoku_action = "Hold"
            if tenkan_sen > kijun_sen:
                ichimoku_action = "Buy"
            elif tenkan_sen < kijun_sen:
                ichimoku_action = "Sell"

            # Итоговое действие
            action = f"MACD: {macd_action} | RSI: {rsi_action} | MA: {ma_action} | Fibonacci: {fib_action} | Volume Profile: {volume_profile_action} | Ichimoku: {ichimoku_action}"

            # Формирование данных
            data = {
                "time": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                "MACD": {
                    "macd_line": float(macd_line[-1]),
                    "signal_line": float(signal_line[-1]),
                    "histogram": float(histogram[-1]),
                    "action": macd_action
                },
                "RSI": {
                    "rsi": float(rsi_values[-1]),
                    "action": rsi_action
                },
                "MovingAverage": {
                    "short_MA": float(ma_short[-1]),
                    "long_MA": float(ma_long[-1]),
                    "action": ma_action
                },
                "Fibonacci": {
                    "levels": {str(level): float(price) for level, price in fib_levels.items()},
                    "action": fib_action
                },
                "VolumeProfile": {
                    "price": profile_price,
                    "VAH": VAH,
                    "VAL": VAL,
                    "ValueArea": value_area,
                    "action": volume_profile_action
                },
                "Ichimoku": {
                    "tenkan_sen": tenkan_sen,
                    "kijun_sen": kijun_sen,
                    "senkou_span_a": senkou_span_a,
                    "senkou_span_b": senkou_span_b,
                    "action": ichimoku_action
                },
                "final_action": action
            }

            results.append(data)

        # Сохранение результата
        with open("indicators_results.json", "w", encoding="utf-8") as file:
            json.dump(results, file, indent=4, ensure_ascii=False)

        print("Данные успешно записаны в indicators_results.json")

    except Exception as e:
        print(f"Ошибка: {e}")

if __name__ == "__main__":
    run_indicators()
