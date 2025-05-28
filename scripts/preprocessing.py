import pandas as pd

# 1. Загрузка данных
df = pd.read_csv('lenta-ru-news_2010_2024.csv')  # укажи правильный путь к файлу

# --- Очистка датасета ---
# 1. Удаляем записи без текста или темы
print(f"До удаления пропусков: {df.shape}")
df = df.dropna(subset=['text', 'topic']).reset_index(drop=True)
print(f"После удаления пропусков: {df.shape}")

# 2. Удаляем темы с количеством статей < 5000
topic_counts = df['topic'].value_counts()
topics_to_keep = topic_counts[topic_counts >= 5000].index

print(f"\nТем, где >= 5000 статей: {len(topics_to_keep)}")

df = df[df['topic'].isin(topics_to_keep)].reset_index(drop=True)
print(f"После удаления редких тем: {df.shape}")

# 3. Проверим финальное распределение тем
print("\nРаспределение статей по темам после чистки:")
print(df['topic'].value_counts())


# --- Ограничение: оставляем по 10k записей для каждой темы ---
df_balanced = df.groupby('topic', group_keys=False).apply(
    lambda x: x.sample(min(len(x), 10000), random_state=42)
).reset_index(drop=True)

print(f"\nФинальный размер сбалансированного датасета: {df_balanced.shape}")
print("Распределение после балансировки (по 10к записей на тему):")
print(df_balanced['topic'].value_counts())

# Сохраняем исправленный датафрейм в новый CSV файл
df_balanced.to_csv('lenta-ru-news_balanced.csv', index=False)

print("Датафрейм сохранен в файл 'lenta-ru-news_balanced.csv'")
