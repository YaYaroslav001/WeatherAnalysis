# Данный скрипт строит для двух точек наблюдения
# графики среднегодовых температур и осадков,
# диаграммы размаха среднегодовых температур и
# декомпозирует данные.

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sys import argv
from statsmodels.tsa.seasonal import seasonal_decompose
from pandas.plotting import register_matplotlib_converters

# Ввод параметров.
try:
    script, data1, data2 = argv
except ValueError:
    print("Недостаточно параметров! Необходимо параметров: 2")
    exit()

# Сохранение названия файлов.
city1 = data1.replace(".txt", "")
city2 = data2.replace(".txt", "")

# Регистрация форматеров и конвертеров pandas в matplotlib.
register_matplotlib_converters()

# Сохранение значения стандартного вывода.
original_stdout = sys.stdout

# Считывание и обработка датасетов.
print("Считывание и обработка данных.")

# Считывание первого датасета.
try:
    data1 = pd.read_csv(data1, sep=";", header=None)
except FileNotFoundError:
    print("Файл " + data1 + " не обнаружен!")
    exit()

# Считывание второго датасета.
try:
    data2 = pd.read_csv(data2, sep=";", header=None)
except FileNotFoundError:
    print("Файл " + data2 + " не обнаружен!")
    exit()

# Присваивание имён столбцам первого датафрейма.
try:
    data1.columns = ["index", "year", "month", "day",
                     "temp_quality", "temp_min", "temp_avg",
                     "temp_max", "precipitation"]
except ValueError:
    print("Количество столбцов первого датасета не равно 9!")
    exit()

# Присваивание имён столбцам второго датафрейма.
try:
    data2.columns = ["index", "year", "month", "day",
                     "temp_quality", "temp_min", "temp_avg",
                     "temp_max", "precipitation"]
except ValueError:
    print("Количество столбцов второго датасета не равно 9!")
    exit()

df1 = pd.DataFrame({'year': data1["year"],      # Заполнение первого датафрейма.
                    'month': data1["month"],
                    'day': data1["day"]})
df1["date"] = pd.to_datetime(df1)
df1["temp_avg"] = data1["temp_avg"]
df1["precipitation"] = pd.to_numeric(data1["precipitation"],
                                     errors='coerce')
df2 = pd.DataFrame({'year': data2["year"],      # Заполнение второго датафрейма.
                    'month': data2["month"],
                    'day': data2["day"]})
df2["date"] = pd.to_datetime(df2)
df2["temp_avg"] = data2["temp_avg"]
df2["precipitation"] = pd.to_numeric(data2["precipitation"],
                                     errors='coerce')

# Стилизация графиков.
sns.set_style("darkgrid")
plt.rc("figure", figsize=(12, 9))
plt.rc("font", size=13)
plt.rc("lines", markersize=5)
plt.rc("lines", linewidth=3)

# Создание папки для хранения результата работы скрипта.
if not os.path.exists("Result"):
    os.makedirs("Result")

# Построения графика среднегодовой температуры города 1.
print("Построения графика среднегодовой температуры (" + city1 + ").")
result1 = df1.groupby('year').mean()    # Группировка первого датафрейма по году.
plt.plot(result1.index, result1['temp_avg'])
plt.title("Среднегодовая температура (" + city1 + ")")
plt.xlabel('Год')
plt.ylabel('Температура (цельсии)')
z = np.polyfit(result1.index, result1['temp_avg'], 1)   # Вычисление линии тренда.
p = np.poly1d(z)
plt.plot(result1.index, p(result1.index), "r--")
plt.savefig('Result/' + city1 + '_Temperature_Plot.png')
plt.clf()

# Построения графика количества осадков за год города 1.
print("Построения графика количества осадков за год (" + city1 + ").")
plt.plot(result1.index, result1['precipitation'])
plt.title("Количество осадков за год (" + city1 + ")")
plt.xlabel('Год')
plt.ylabel('Количество осадков')
z = np.polyfit(result1.index, result1['precipitation'], 1)  # Вычисление линии тренда.
p = np.poly1d(z)
plt.plot(result1.index, p(result1.index), "r--")
plt.savefig('Result/' + city1 + '_Precipitations_Plot.png')
plt.clf()

# Построения графика среднегодовой температуры города 2.
print("Построения графика среднегодовой температуры (" + city2 + ").")
result2 = df2.groupby('year').mean()    # Группировка второго датафрейма по году.
plt.plot(result2.index, result2['temp_avg'])
plt.title("Среднегодовая температура (" + city2 + ")")
plt.xlabel('Год')
plt.ylabel('Температура (цельсии)')
z = np.polyfit(result2.index, result2['temp_avg'], 1)   # Вычисление линии тренда.
p = np.poly1d(z)
plt.plot(result2.index, p(result2.index), "r--")
plt.savefig('Result/' + city2 + '_Temperature_Plot.png')
plt.clf()

# Построения графика количества осадков за год города 2.
print("Построения графика количества осадков за год (" + city2 + ").")
plt.plot(result2.index, result2['precipitation'])
plt.title("Количество осадков за год (" + city2 + ")")
plt.xlabel('Год')
plt.ylabel('Количество осадков')
z = np.polyfit(result2.index, result2['precipitation'], 1)  # Вычисление линии тренда.
p = np.poly1d(z)
plt.plot(result2.index, p(result2.index), "r--")
plt.savefig('Result/' + city2 + '_Precipitations_Plot.png')
plt.clf()

# Построения графика корреляции среднегодовой температуры двух городов.
print("Построения графика среднегодовой температуры двух точек наблюдения.")
plt.plot(result1.index, result1['temp_avg'],
         label="Среднегодовая температура (" + city1 + ")")
plt.plot(result2.index, result2['temp_avg'],
         label="Среднегодовая температура (" + city2 + ")")
plt.title("Среднегодовая температура в " + city1 + " и " + city2)
plt.xlabel('Год')
plt.ylabel('Температура (цельсии)')
plt.legend()
plt.savefig('Result/' + city1 + '_' + city2 + '_Temperature_Plot.png')
plt.clf()

# Построения графика корреляции количества осадков за год двух городов.
print("Построения графика количества осадков за год двух точек наблюдения.")
plt.plot(result1.index, result1['precipitation'],
         label="Среднегодовые осадки (" + city1 + ")")
plt.plot(result2.index, result2['precipitation'],
         label="Среднегодовые осадки (" + city2 + ")")
plt.title("Среднегодовые осадки в " + city1 + " и " + city2)
plt.xlabel('Год')
plt.ylabel('Количество осадков')
plt.legend()
plt.savefig('Result/' + city1 + '_' + city2 + '_Precipitations_Plot.png')
plt.clf()

# Построения диаграммы размаха среднегодовой температуры города 1.
print("Построения диаграммы размаха среднегодовой температуры (" + city1 + ").")
sns.boxplot(data=df1, x='month', y='temp_avg')
plt.xlabel('Месяц')
plt.ylabel('Температура (цельсии)')
plt.title('Температура (' + city1 + ')')
plt.savefig('Result/' + city1 + '_Temperature_Boxplot.png')
plt.clf()

# Построения диаграммы размаха среднегодовой температуры города 2.
print("Построения диаграммы размаха среднегодовой температуры (" + city2 + ").")
sns.boxplot(data=df2, x='month', y='temp_avg')
plt.xlabel('Месяц')
plt.ylabel('Температура (цельсии)')
plt.title('Температура (' + city2 + ')')
plt.savefig('Result/' + city2 + '_Temperature_Boxplot.png')
plt.clf()

# Декомпозиция данных по температуре города 1.
print("Декомпозиция данных по температуре (" + city1 + ").")
result = seasonal_decompose(result1['temp_avg'], model='additive', period=12)
result.plot()
plt.savefig('Result/' + city1 + '_Temperature_Decomposition.png')
plt.clf()

# Декомпозиция данных по осадкам города 1.
print("Декомпозиция данных по осадкам (" + city1 + ").")
result = seasonal_decompose(result1['precipitation'], model='additive', period=12)
result.plot()
plt.savefig('Result/' + city1 + '_Precipitations_Decomposition.png')
plt.clf()

# Декомпозиция данных по температуре города 2.
print("Декомпозиция данных по температуре (" + city2 + ").")
result = seasonal_decompose(result2['temp_avg'], model='additive', period=12)
result.plot()
plt.savefig('Result/' + city2 + '_Temperature_Decomposition.png')
plt.clf()

# Декомпозиция данных по осадкам города 2.
print("Декомпозиция данных по осадкам (" + city2 + ").")
result = seasonal_decompose(result2['precipitation'], model='additive', period=12)
result.plot()
plt.savefig('Result/' + city2 + '_Precipitations_Decomposition.png')
plt.clf()

print("\nРабота успешно завершена!")