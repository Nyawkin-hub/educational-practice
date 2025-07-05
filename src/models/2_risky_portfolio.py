import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from itertools import combinations

# Загрузка данных
data = pd.read_csv('portfolio_prices.csv', index_col=0)
companies = data.columns.tolist()

returns = data.pct_change().dropna()
expected_returns = returns.mean()
cov_matrix = returns.cov()

def portfolio_risk(weights, cov):
    return np.sqrt(np.dot(weights.T, np.dot(cov, weights)))

def portfolio_return(weights, returns_mean):
    return np.dot(weights, returns_mean)

def optimize_portfolio_pair(returns_mean, cov, target_return):
    constraints = (
        {"type": "eq", "fun": lambda x: np.sum(x) - 1},
        {"type": "eq", "fun": lambda x: portfolio_return(x, returns_mean) - target_return},
    )
    bounds = [(0, 1), (0, 1)]
    init_weights = np.array([0.5, 0.5])
    
    result = minimize(
        portfolio_risk,
        init_weights,
        args=(cov,),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    if result.success:
        return result.x
    else:
        return None

all_results = []

# Перебор всех пар активов
for asset1, asset2 in combinations(companies, 2):
    pair_returns = returns[[asset1, asset2]]
    pair_expected_returns = pair_returns.mean()
    pair_cov = pair_returns.cov()
    
    # Диапазон целевых доходностей между минимальной и максимальной доходностью пары
    target_returns = np.linspace(pair_expected_returns.min(), pair_expected_returns.max(), 50)
    
    pair_portfolios = []
    for target in target_returns:
        weights = optimize_portfolio_pair(pair_expected_returns.values, pair_cov.values, target)
        if weights is not None:
            ret = portfolio_return(weights, pair_expected_returns.values)
            risk = portfolio_risk(weights, pair_cov.values)
            pair_portfolios.append({
                "assets": (asset1, asset2),
                "weights": weights,
                "return": ret,
                "risk": risk
            })
    
    all_results.extend(pair_portfolios)

# Конвертируем в DataFrame для удобства анализа
df_results = pd.DataFrame(all_results)

# Ищем глобальный портфель с минимальным риском
min_risk = df_results['risk'].min()
min_risk_portfolios = df_results[np.isclose(df_results['risk'], min_risk, atol=1e-6)]
optimal_portfolio = min_risk_portfolios.loc[min_risk_portfolios['return'].idxmax()]

# Визуализация
plt.figure(figsize=(14, 9))

# Рисуем все линии эффективных границ по парам
for (asset1, asset2), group in df_results.groupby('assets'):
    plt.plot(group['risk'], group['return'], label=f'{asset1} & {asset2}', alpha=0.3)
    
    # Находим минимальный риск для этой пары
    min_risk_in_pair = group['risk'].min()
    min_risk_portfolios_in_pair = group[np.isclose(group['risk'], min_risk_in_pair, atol=1e-6)]
    
    # Отмечаем красными точками минимальные риски в паре
    plt.scatter(min_risk_portfolios_in_pair['risk'], min_risk_portfolios_in_pair['return'], color='red', s=80)

# Отмечаем глобальный минимальный риск крупной красной точкой с подписью
plt.scatter(optimal_portfolio['risk'], optimal_portfolio['return'], color='darkred', s=150, label='Мин. риск (оптимальный)')

plt.xlabel('Риск (σ)')
plt.ylabel('Доходность (E)')
plt.title('Эффективные границы для всех пар активов')
plt.grid(True)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

# Подпись к глобальному оптимальному портфелю
w1, w2 = optimal_portfolio['weights']
a1, a2 = optimal_portfolio['assets']
plt.text(optimal_portfolio['risk'], optimal_portfolio['return'],
         f'\n{a1}: {w1:.1%}\n{a2}: {w2:.1%}',
         fontsize=10, color='darkred',
         verticalalignment='bottom', horizontalalignment='right')

plt.tight_layout()
plt.show()