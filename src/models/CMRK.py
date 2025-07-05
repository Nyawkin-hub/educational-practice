import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from matplotlib.widgets import Cursor

# --- 1. Загрузка данных ---
data = pd.read_csv('portfolio_prices.csv', index_col=0)

companies = data.columns.tolist()
returns = data.pct_change().dropna()
expected_returns = returns.mean()
cov_matrix = returns.cov()

# --- 2. Безрисковая ставка ---
r_f = 0.0001  # 0.01% дневная доходность (можно заменить на свою)

# --- 3. Функции для портфеля ---
def portfolio_return(weights):
    return np.dot(weights, expected_returns)

def portfolio_risk(weights):
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

# --- 4. Найти тангенциальный портфель (max Sharpe ratio) ---
def neg_sharpe(weights):
    ret = portfolio_return(weights)
    risk = portfolio_risk(weights)
    if risk == 0:
        return 1e6  # Защита от деления на 0
    return -(ret - r_f) / risk

constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
bounds = [(0, 1) for _ in companies]
init_weights = np.ones(len(companies)) / len(companies)

result = minimize(neg_sharpe, init_weights, method='SLSQP', bounds=bounds, constraints=constraints)
tangential_weights = result.x
tangential_return = portfolio_return(tangential_weights)
tangential_risk = portfolio_risk(tangential_weights)
max_sharpe = (tangential_return - r_f) / tangential_risk

print(f"Максимальный коэффициент Шарпа: {max_sharpe:.4f}")

# --- 5. Построение эффективной границы ---
target_returns = np.linspace(expected_returns.min(), expected_returns.max(), 100)

def optimize_portfolio(target_ret):
    cons = (
        {"type": "eq", "fun": lambda x: np.sum(x) - 1},
        {"type": "eq", "fun": lambda x: portfolio_return(x) - target_ret}
    )
    res = minimize(portfolio_risk, init_weights, method='SLSQP', bounds=bounds, constraints=cons)
    return res.x if res.success else None

portfolios = []
for ret in target_returns:
    w = optimize_portfolio(ret)
    if w is not None:
        portfolios.append({
            "weights": w,
            "return": portfolio_return(w),
            "risk": portfolio_risk(w)
        })

results = pd.DataFrame([{
    "return": p["return"],
    "risk": p["risk"],
    **{c: w for c, w in zip(companies, p["weights"])}
} for p in portfolios])

# --- 6. Линия рынка капитала (CML) ---
cml_risk = np.linspace(0, results["risk"].max() * 1.2, 200)
cml_return = r_f + max_sharpe * cml_risk

# --- 7. Визуализация ---
fig, ax = plt.subplots(figsize=(12, 8))
scatter = ax.scatter(results["risk"], results["return"], c=results["return"], cmap="viridis", picker=True)
plt.colorbar(scatter, label="Доходность")
plt.xlabel("Риск (σ)")
plt.ylabel("Доходность (E)")
plt.title("Эффективная граница и линия рынка капитала (CML)")
plt.grid(True)

# Курсор
cursor = Cursor(ax, useblit=True, color='red', linewidth=1)

# Безрисковый актив
ax.scatter(0, r_f, color='green', s=100, label='Безрисковый актив')

# Тангенциальный портфель
ax.scatter(tangential_risk, tangential_return, color='red', s=150, label='Тангенциальный портфель')

# Линия рынка капитала (CML)
ax.plot(cml_risk, cml_return, linestyle='--', color='orange', label='Линия рынка капитала (CML)')

ax.legend()

def on_pick(event):
    ind = event.ind[0]
    portfolio = results.iloc[ind]
    plt.figure(figsize=(10, 5))
    plt.suptitle(f"Портфель: Доходность = {portfolio['return']:.2%}, Риск = {portfolio['risk']:.2%}")
    plt.subplot(1, 2, 1)
    weights = portfolio[companies]
    weights = weights[weights > 0.001]  # фильтр для отображения значимых долей >0.1%
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

# --- 9. Сохранение результатов ---
results.to_csv('portfolio_results_cml.csv', index=False)
