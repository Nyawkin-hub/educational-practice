import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from matplotlib.widgets import Cursor

# Загрузка данных
data = pd.read_csv('portfolio_prices.csv', index_col=0)

companies = data.columns.tolist()
returns = data.pct_change().dropna()  # дневные доходности
expected_returns = returns.mean()
cov_matrix = returns.cov()
vols = returns.std()

# Определяем безрисковый актив (риск < 1e-6), если есть — выбираем с макс доходностью
riskless_threshold = 1e-6
riskless_assets = vols[vols < riskless_threshold]

if not riskless_assets.empty:
    best_riskless_asset = riskless_assets.index[np.argmax(expected_returns[riskless_assets.index])]
    r_f = expected_returns[best_riskless_asset]
else:
    r_f = 0.0001  # фиксированная безрисковая ставка (0.01% в день)

print(f"Безрисковая ставка r_f = {r_f:.6f}")

# Функции для одного рискованного актива и безрискового
def portfolio_return(w, r_i):
    # w - доля рискованного актива, (1-w) - безрисковый
    return w * r_i + (1 - w) * r_f

def portfolio_risk(w, sigma_i):
    # риск только от рискованного актива
    return w * sigma_i

# Собираем результаты для каждого рискованного актива
results = []

for asset in companies:
    # Пропускаем безрисковый актив, если он есть в выборке
    if asset == riskless_assets.index[0] if not riskless_assets.empty else None:
        continue
    
    r_i = expected_returns[asset]
    sigma_i = vols[asset]

    # Диапазон весов в рискованном активе от 0 до 1
    weights = np.linspace(0, 1, 100)

    for w in weights:
        ret = portfolio_return(w, r_i)
        risk = portfolio_risk(w, sigma_i)
        results.append({
            'asset': asset,
            'weight_risky': w,
            'return': ret,
            'risk': risk
        })

df_results = pd.DataFrame(results)

# Поиск оптимального рискованного актива — максимальный Sharpe (возьмём w=1)
sharpe_ratios = (expected_returns - r_f) / vols
if not riskless_assets.empty:
    sharpe_ratios = sharpe_ratios.drop(best_riskless_asset, errors='ignore')

best_asset = sharpe_ratios.idxmax()
max_sharpe = sharpe_ratios.max()
best_risky_sigma = vols[best_asset]
best_risky_return = expected_returns[best_asset]

print(f"Оптимальный рискованный актив: {best_asset} с коэффициентом Шарпа = {max_sharpe:.4f}")

# Линия рынка капитала (CML) от безрисковой ставки до максимально рискованного портфеля
cml_risk = np.linspace(0, best_risky_sigma * 1.2, 200)
cml_return = r_f + max_sharpe * cml_risk

# Визуализация
fig, ax = plt.subplots(figsize=(12, 8))

# График для каждого рискованного актива
for asset, group in df_results.groupby('asset'):
    ax.plot(group['risk'], group['return'], alpha=0.3, label=asset)

# Отметим безрисковый актив
ax.scatter(0, r_f, color='green', s=100, label='Безрисковый актив')

# Отметим оптимальный рискованный актив с w=1
ax.scatter(best_risky_sigma, best_risky_return, color='red', s=150, label=f'Оптимальный рискованный актив: {best_asset}')

# Линия рынка капитала
ax.plot(cml_risk, cml_return, linestyle='--', color='orange', label='Линия рынка капитала (CML)')

plt.xlabel('Риск (σ)')
plt.ylabel('Доходность (E)')
plt.title('Оптимальный портфель: Безрисковый + один рискованный актив')
plt.grid(True)
plt.legend()

cursor = Cursor(ax, useblit=True, color='red', linewidth=1)

# Интерактивный клик по точкам
def on_pick(event):
    ind = event.ind[0]
    row = df_results.iloc[ind]
    w = row['weight_risky']
    asset = row['asset']
    ret = row['return']
    risk = row['risk']

    plt.figure(figsize=(8, 4))
    plt.suptitle(f"Портфель из безрискового и {asset}\nДоходность = {ret:.2%}, Риск = {risk:.2%}")
    plt.pie([w, 1 - w], labels=[asset, 'Безрисковый'], autopct='%1.1f%%')
    plt.show()

scatter = ax.scatter(df_results['risk'], df_results['return'], c=df_results['return'], cmap='viridis', picker=True)

fig.canvas.mpl_connect('pick_event', on_pick)

plt.show()
