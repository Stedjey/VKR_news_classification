# --- Импорт библиотек ---
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score
from scipy.special import softmax
from transformers import EarlyStoppingCallback


# --- 1️⃣ Подготовка данных ---

texts = df_balanced['text']
labels = df_balanced['topic']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

# Кодируем метки
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)

num_classes = len(le.classes_)

# --- 2️⃣ Токенизация для BERT ---

model_name = "DeepPavlov/rubert-base-cased"

tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=256)

# Подготовка данных для Hugging Face Dataset
train_data = pd.DataFrame({"text": X_train.tolist(), "label": y_train_enc})
test_data = pd.DataFrame({"text": X_test.tolist(), "label": y_test_enc})

train_ds = Dataset.from_pandas(train_data)
test_ds = Dataset.from_pandas(test_data)

train_ds = train_ds.map(tokenize, batched=True)
test_ds = test_ds.map(tokenize, batched=True)

# Убираем колонки с текстом
train_ds = train_ds.remove_columns(["text"])
test_ds = test_ds.remove_columns(["text"])

train_ds = train_ds.with_format("torch")
test_ds = test_ds.with_format("torch")

# --- 3️⃣ Создание модели ---

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_classes
)

# --- 4️⃣ Метрики для Trainer ---

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)

    accuracy = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average='macro')
    weighted_f1 = f1_score(labels, preds, average='weighted')

    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1
    }

# --- 5️⃣ Параметры обучения ---

training_args = TrainingArguments(
    output_dir="./rubert_results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir="./logs",
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1"
)

# --- 6️⃣ Тренировка модели ---

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

trainer.train()

# --- 6.1️⃣ Сохранение модели и Tokenizer ---

trainer.save_model("./rubert_final_model")
tokenizer.save_pretrained("./rubert_final_model")

# Сохраняем LabelEncoder
import pickle
with open("rubert_label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print("Модель, Tokenizer и LabelEncoder сохранены успешно!")

# --- 7️⃣ Предсказания и метрики ---

predictions = trainer.predict(test_ds)
y_pred_enc = np.argmax(predictions.predictions, axis=1)
y_pred = le.inverse_transform(y_pred_enc)
y_true = y_test.tolist()

print("Classification Report:")
print(classification_report(y_true, y_pred))

accuracy = accuracy_score(y_true, y_pred)
macro_f1 = f1_score(y_true, y_pred, average='macro')
weighted_f1 = f1_score(y_true, y_pred, average='weighted')

# ROC AUC

y_test_bin = label_binarize(y_test_enc, classes=np.arange(len(le.classes_)))
y_pred_probs = softmax(predictions.predictions, axis=1)
roc_auc = roc_auc_score(y_test_bin, y_pred_probs, average='macro')

print(f"\nОсновные метрики:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Macro F1-score: {macro_f1:.4f}")
print(f"Weighted F1-score: {weighted_f1:.4f}")
print(f"ROC AUC (macro): {roc_auc:.4f}")

# --- 8️⃣ Матрица ошибок ---

cm = confusion_matrix(y_true, y_pred, labels=le.classes_)

plt.figure(figsize=(14, 12))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=le.classes_,
    yticklabels=le.classes_
)
plt.ylabel('Истинный класс')
plt.xlabel('Предсказанный класс')
plt.title('Матрица ошибок (RuBERT)')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()