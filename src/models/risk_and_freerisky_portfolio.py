import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from matplotlib.widgets import Cursor

# Загрузка данных
data = pd.read_csv('portfolio_prices.csv', index_col=0)

companies = data.columns.tolist()
returns = data.pct_change().dropna()  # доходности в день
expected_returns = returns.mean()
cov_matrix = returns.cov()

vols = returns.std()  # волатильность
# Считаем, что актив "безрисковый", если его дневной риск меньше 1e-6
riskless_threshold = 1e-6
riskless_assets = vols[vols < riskless_threshold]

if not riskless_assets.empty:
    best_riskless_asset = riskless_assets.index[np.argmax(expected_returns[riskless_assets.index])]
    r_f = expected_returns[best_riskless_asset]
else:
    # если нет безрисковых активов — фиксированая ставка
    r_f = 0.0001  # 0.01% дневная доходность

# Расчет риска
def portfolio_risk(weights):
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

# расчет доходности
def portfolio_return(weights):
    return np.dot(weights, expected_returns)

# эффективная граница
def optimize_portfolio(target_return):
    constraints = (
        {"type": "eq", "fun": lambda x: np.sum(x) - 1},  # сумма весов = 1
        {"type": "eq", "fun": lambda x: portfolio_return(x) - target_return},  # доходность = target_return
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
    return result.x if result.success else None

# 
target_returns = np.linspace(0, expected_returns.max(), 150)

# эффективной граница
portfolios = []
for ret in target_returns:
    weights = optimize_portfolio(ret)
    if weights is not None:
        port_ret = portfolio_return(weights)
        port_risk = portfolio_risk(weights)
        portfolios.append({"weights": weights, "return": port_ret, "risk": port_risk})

results = pd.DataFrame([{
    "return": p["return"],
    "risk": p["risk"],
    **{ticker: weight for ticker, weight in zip(companies, p["weights"])}
} for p in portfolios])

# тангенциальный портфель с максимальным кфом шарпа
def neg_sharpe(weights):
    ret = portfolio_return(weights)
    risk = portfolio_risk(weights)
    if risk == 0:
        return 1e6  # защита от деления на 0
    return -(ret - r_f) / risk

constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
bounds = [(0, 1) for _ in companies]
init_weights = np.ones(len(companies)) / len(companies)

result = minimize(neg_sharpe, init_weights, method="SLSQP", bounds=bounds, constraints=constraints)
tangential_weights = result.x
tangential_return = portfolio_return(tangential_weights)
tangential_risk = portfolio_risk(tangential_weights)
max_sharpe = (tangential_return - r_f) / tangential_risk

# Построение линии рынка капитала (CML)
cml_risks = np.linspace(0, results["risk"].max()*1.2, 200)
cml_returns = r_f + max_sharpe * cml_risks

# Визуализация
fig, ax = plt.subplots(figsize=(12, 8))
scatter = ax.scatter(results["risk"], results["return"], c=results["return"], cmap="viridis", picker=True)
plt.colorbar(scatter, label="Доходность")
plt.xlabel("Риск (σ)")
plt.ylabel("Доходность (E)")
plt.title("Эффективная граница и линия рынка капитала (CML)")
plt.grid(True)

# Курсор
cursor = Cursor(ax, useblit=True, color='red', linewidth=1)

# безриск актив
ax.scatter(0, r_f, color='green', s=100, label='Безрисковый актив')

# тангенциальный портфель
ax.scatter(tangential_risk, tangential_return, color='red', s=150, label='Тангенциальный портфель')

# CML
ax.plot(cml_risks, cml_returns, linestyle='--', color='orange', label='Линия рынка капитала (CML)')

ax.legend()

def on_pick(event):
    ind = event.ind[0]
    portfolio = results.iloc[ind]
    plt.figure(figsize=(10, 5))
    plt.suptitle(f"Портфель: Доходность = {portfolio['return']:.2%}, Риск = {portfolio['risk']:.2%}")
    plt.subplot(1, 2, 1)
    weights = portfolio[companies]
    weights = weights[weights > 0.001]  # > 0.1%
    plt.pie(weights, labels=weights.index, autopct='%1.1f%%')
    plt.title("Состав портфеля")
    plt.subplot(1, 2, 2)
    info_text = "\n".join([f"{ticker}: {weight:.1%}" for ticker, weight in weights.items()])
    plt.text(0.1, 0.5, info_text, fontsize=10)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

fig.canvas.mpl_connect('pick_event', on_pick)

plt.show()

results.to_csv('portfolio_results.csv', index=False)
