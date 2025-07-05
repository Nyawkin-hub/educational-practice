import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import linprog
from matplotlib.widgets import Cursor

# Загрузка данных и расчет доходностей
data = pd.read_csv('portfolio_prices.csv', index_col=0)
returns = data.pct_change().dropna()
expected_returns = returns.mean()
cov_matrix = returns.cov()

companies = data.columns.tolist()
n = len(companies)

# Ограничения
max_weight = 0.3  # Максимум на один актив (например 30%)
bounds = [(0, max_weight) for _ in range(n)]

# Цель: максимизировать доходность => минимизируем -доходность
c = -expected_returns.values

# Ограничение суммы весов == 1
A_eq = np.ones((1, n))
b_eq = np.array([1])

# Решение задачи линейного программирования
res = linprog(c=c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

if res.success:
    weights = res.x
    port_return = np.dot(weights, expected_returns)
    port_risk = np.sqrt(weights.T @ cov_matrix.values @ weights)
    print("Оптимальные веса портфеля:")
    for ticker, w in zip(companies, weights):
        print(f"{ticker}: {w:.4f}")
    print(f"Ожидаемая доходность портфеля: {port_return:.4%}")
    print(f"Риск портфеля (стандартное отклонение): {port_risk:.4%}")
else:
    print("Оптимизация не удалась")

# Визуализация: случайные портфели и оптимальный портфель
num_samples = 5000
random_weights = np.random.dirichlet(np.ones(n), num_samples)
random_returns = random_weights @ expected_returns.values
random_risks = np.sqrt(np.einsum('ij,jk,ik->i', random_weights, cov_matrix.values, random_weights))

fig, ax = plt.subplots(figsize=(12, 8))
scatter_random = ax.scatter(random_risks, random_returns, c='lightgray', alpha=0.5, label='Случайные портфели')
scatter_opt = ax.scatter(port_risk, port_return, color='red', s=150, label='Оптимальный портфель ЛП', picker=5)

plt.xlabel('Риск (стандартное отклонение)')
plt.ylabel('Ожидаемая доходность')
plt.title('Оптимальный портфель по линейному программированию')
plt.legend()
plt.grid(True)

cursor = Cursor(ax, useblit=True, color='blue', linewidth=1)

def on_pick(event):
    # Проверяем, что выбрана именно точка оптимального портфеля
    if event.artist != scatter_opt:
        return
    # Выводим состав портфеля
    plt.figure(figsize=(10, 5))
    plt.suptitle("Оптимальный портфель")
    weights_nonzero = {ticker: w for ticker, w in zip(companies, weights) if w > 1e-4}
    plt.subplot(1, 2, 1)
    plt.pie(list(weights_nonzero.values()), labels=list(weights_nonzero.keys()), autopct='%1.1f%%')
    plt.title("Состав портфеля")
    plt.subplot(1, 2, 2)
    info_text = "\n".join([f"{ticker}: {weight:.2%}" for ticker, weight in weights_nonzero.items()])
    plt.text(0.1, 0.5, info_text, fontsize=12)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

fig.canvas.mpl_connect('pick_event', on_pick)
plt.show()
