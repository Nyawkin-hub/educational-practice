import requests
import pathlib
import pandas as pd
from apimoex import get_board_history
from datetime import datetime
from tqdm import tqdm

board = 'TQBR'                  # рынок
start = datetime(2023, 6, 1)    # TRADEDATE начала котировок

# качает котировки за все годы а не так как надо
# TODO исправить 
end = datetime(2025, 6, 1)      # TRADEDATE конца котировок

# вытаскиваем индексы компаний из файла companies
with open("./companies.txt", "r") as f:    
    companies = [line.strip() for line in f]

# путь для сохранения
output_dir = pathlib.Path(f"./src/data/{board}") 
output_dir.mkdir(parents=True, exist_ok=True)

# # сессия для реквестов на moex
# with requests.Session() as session:
#     for company in tqdm(companies):
#         try: # на случай ошибки
#             data = get_board_history(session, 
#                                      company, 
#                                      board=board, 
#                                      start=start, 
#                                      end=end)
#             if not data:
#                 print(f"[!] Нет данных: {company}")                                              # TODO логи
#                 continue
#             df = pd.DataFrame(data)[['TRADEDATE', 'CLOSE']] # что будет в csv файлах
#             df.to_csv(output_dir / f"{company}.csv", index=False)
#         except Exception as e:
#             print(f"[ERROR] {company}: {e}")                                                     # TODO логи

# merge + rename
master_df = pd.DataFrame()
for company in tqdm(companies):
    file_path = f"./src/data/TQBR/{company}.csv"
    try:
        df = pd.read_csv(
            file_path,
            usecols=['TRADEDATE', 'CLOSE'],
            parse_dates=['TRADEDATE']
        )
        df = df.rename(columns={'CLOSE': company})
        if master_df.empty:
            master_df = df
        else:
            master_df = pd.merge(master_df, df, on='TRADEDATE', how='outer')
            
    except Exception as e:
        print(f"Ошибка при обработке {company}: {e}")

try:
    master_df = master_df.sort_values('TRADEDATE').reset_index(drop=True)
    master_df.to_csv('portfolio_prices.csv', index=False)
    print("Результат сохранён в 'portfolio_prices.csv'")
except Exception as e:
    print(f"Ошибка при обработке master_df: {e}")

# фильтр по TRADEDATE
start_date = '2023-07-01'
end_date = '2025-07-01'
master_df = master_df.sort_values('TRADEDATE').reset_index(drop=True)
filtered_df = master_df[
    (master_df['TRADEDATE'] >= start_date) & 
    (master_df['TRADEDATE'] <= end_date)
]

# вычеркивает компании у которых % пропусков большой + строки дат с пропусками для портфеля
max_missing_stats = 10
missing_stats = filtered_df.drop(columns='TRADEDATE').isnull().mean() * 100
missing_stats = missing_stats.round(2).sort_values(ascending=False)
print("Процент пропусков по компаниям:")
print(missing_stats)
valid_tickers = missing_stats[missing_stats <= max_missing_stats].index.tolist()
print("\nКомпании, удовлетворяющие критерию (<={max_missing_stats}% пропусков):")
print(valid_tickers)
final_df = filtered_df[['TRADEDATE'] + valid_tickers]
final_df = final_df.dropna()
output_file = 'portfolio_prices.csv'
final_df.to_csv(output_file, index=False)
print(f"Финальный результат сохранен в {output_file}")
print(f"Период данных: {start_date} - {end_date}")
print(f"Количество компаний: {len(valid_tickers)}")
print(f"Количество дней: {len(final_df)}")