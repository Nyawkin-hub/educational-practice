import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from matplotlib.widgets import Cursor

# Загрузка данных
data = pd.read_csv('portfolio_prices.csv', index_col=0)

companies = data.columns.tolist()[0:]
returns = data.pct_change().dropna() # доход в день 
expected_returns = returns.mean()
cov_matrix = returns.cov()

# расчета риска 
def portfolio_risk(weights):
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

# расчета доходности
def portfolio_return(weights):
    return np.dot(weights, expected_returns)

# оптимизация портфеля
def optimize_portfolio(target_return):
    constraints = (
        {"type": "eq", "fun": lambda x: np.sum(x) - 1},     # сумма весов = 1
        {"type": "eq", "fun": lambda x: portfolio_return(x) - target_return}, # заданная доходность
    )
    bounds = [(0, 1) for _ in companies]
    init_weights = np.ones(len(companies)) / len(companies)
    
    result = minimize(
        portfolio_risk,
        init_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    return result.x

# target_returns = np.linspace(expected_returns.min(), expected_returns.max(), 150) # с отрицательным доходом
target_returns = np.linspace(0, expected_returns.max(), 100) # без отрицательного

# расчет портфелей
portfolios = []
for target in target_returns:
    weights = optimize_portfolio(target)
    ret = portfolio_return(weights)
    risk = portfolio_risk(weights)
    portfolios.append({"weights": weights, "return": ret, "risk": risk})

# сбор результатов
results = pd.DataFrame([{
    "return": p["return"],
    "risk": p["risk"],
    **{ticker: weight for ticker, weight in zip(companies, p["weights"])}
} for p in portfolios])

# портфель с минимальным риском
min_risk_value = results["risk"].min()
min_risk_candidates = results[np.isclose(results["risk"], min_risk_value, atol=1e-6)]
true_min_risk_portfolio = min_risk_candidates.loc[min_risk_candidates["return"].idxmax()]

# график
fig, ax = plt.subplots(figsize=(12, 8))
scatter = ax.scatter(results["risk"], results["return"], c=results["return"], cmap="viridis", picker=True)
plt.colorbar(scatter, label="Доходность")
plt.xlabel("Риск (σ)")
plt.ylabel("Доходность (E)")
plt.title("")
plt.grid(True)

# курсор (красный)
cursor = Cursor(ax, useblit=True, color='red', linewidth=1)

ax.scatter(
    true_min_risk_portfolio["risk"],
    true_min_risk_portfolio["return"],
    color='red',
    s=100,
    label='Мин. риск (опт.)'
)
ax.legend()

# клик
def on_pick(event):
    ind = event.ind[0]
    portfolio = results.iloc[ind]
    
    # новое окно с информацией
    plt.figure(figsize=(10, 5))
    plt.suptitle(f"Портфель: Доходность = {portfolio['return']:.2%}, Риск = {portfolio['risk']:.2%}")
    
    # график состава портфеля
    plt.subplot(1, 2, 1)
    weights = portfolio[companies]
    weights = weights[weights > 0.001]  # активы с долей > 0.1%
    plt.pie(weights, labels=weights.index, autopct='%1.1f%%')
    plt.title("Состав портфеля")
    
    # текст
    plt.subplot(1, 2, 2)
    info_text = "\n".join([f"{ticker}: {weight:.1%}" 
                          for ticker, weight in weights.items()])
    plt.text(0.1, 0.5, info_text, fontsize=10)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# подключение обработчика событий
fig.canvas.mpl_connect('pick_event', on_pick)

plt.show()

results.to_csv('portfolio_results.csv', index=False)