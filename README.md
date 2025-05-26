# VKR_news_classification

VKR_news_classification/
│
├── app/                            # Основной код системы
│   ├── bot.py                      # Telegram-бот для классификации текстов
│   ├── streamlit_app.py           # Веб-интерфейс на Streamlit
│   ├── create_db.py               # Скрипт создания SQLite-баз данных
│
├── notebooks/                     # Jupyter-ноутбуки с обучением моделей и анализами
│   ├── 1_base_models.ipynb
│   ├── 2_neural_models.ipynb
│   ├── 3_transformers.ipynb
│   ├── 4_topic_modeling.ipynb    
│   └── 5_sentiment_analysis.ipynb
│
├── rubert_final_model/            # Весы и конфигурация дообученного RuBERT
│   ├── config.json
│   ├── model.safetensors          
│   ├── tokenizer_config.json
│   └── и другие файлы модели
│
├── parse_lenta_ru.py              # Скрипт для парсинга новостей с Lenta.ru
├── rubert_label_encoder.pkl       # Сериализованный LabelEncoder (joblib/pickle)
├── requirements.txt               # Python-зависимости проекта
├── Dockerfile                     # Сборка контейнера
├── docker-compose.yml             # Запуск системы в одном контейнере
├── compose.sh                     # Удобный скрипт запуска Docker Compose
└── README.md                      
