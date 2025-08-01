# PyTorch Model Training Framework

Этот проект демонстрирует удобный и универсальный подход к обучению моделей в PyTorch через единый класс, который берет на себя:

- Подготовку и разбиение датасета  
- Обучение, валидацию и тестирование  
- Сохранение и загрузку чекпоинтов с состоянием модели, оптимизатора и шедулера  
- Логирование метрик и построение графиков обучения  
- Поддержку как классификации, так и регрессии  

Основной упор проекта сделан на структуру и оформление кода, чтобы обучение новых моделей было быстрым, чистым и воспроизводимым.

---

## 📂 Структура проекта
```python
├── models/ # чекпоинты из примера
├── src/ # Классы и функции проекта
├── classification.ipynb # Пример Классификации на датасете MNIST
├── regression.ipynb # Пример Реграссии на синтетическом датасете
├── dataset_generator.py # Скрипт для формирования датасетов
├── requirements.txt # зависимости проекта
└── README.md
```

> Датасеты генерируются скриптом в корне папки по умолчанию, но путь можно поменять изменив переменную `PATH`

---

## 🚀 Основные возможности

Класс `MyModel` реализует полный цикл обучения модели:

- Разбиение датасета на `train/val/test` по списку долей  
- Гибкая настройка параметров через аргументы конструктора  
- Поддержка ранней остановки (`earlystoper`)
- Загрузка сохранённых состояний и продолжение обучения
- Автоматическая визуализация метрик (loss и accuracy)
- Сохранение чекпоинтов в виде словаря:

```python
checkpoint = {
    'state_model': model.state_dict(),
    'state_optimizer': optimizer.state_dict(),
    'state_scheduler': scheduler.state_dict(),
    'loss': {
        'train_loss': train_loss_hist,
        'val_loss': val_loss_hist,
        'best loss': best_loss
    },
    'metric': {
        'train_acc': train_acc_hist,
        'val_acc': val_acc_hist,
        'Ir': lr_hist
    },
    'save_epoch': step
}
```

⚙️ Аргументы класса:

- `dataset` – датасет для обучения
- `model` – PyTorch модель
- `optimizer` – оптимизатор
- `loss_function` – функция потерь
- `sheduler` – шедулер learning rate
- `save_path` – путь сохранения чекпоинтов
- `name` – имя проекта/модели
- `classification` – переключение режима классификации и регрессии
- `is_reshape` - включение решейпа входных тензоров
- `earlystoper` – объект ранней остановки
- `split_list` – доли для train/val/test
- `batch` - размер батча
- `save_trashold` - порог чувствительности к значению ошибки необходимый для автоматического сохранения чекпоинта
- `plot` – визуализация графиков обучения
- `device` - использование GPU/CPU
- `offload` - выгрузка модели с GPU

# 📊 Результаты
#### MNIST Classification
- Mean Loss: 0.0398
- Accuracy: 0.9921

#### Square Center Regression
- Mean Loss: 0.0235
- Accuracy: 0.9963

# 💡 Преимущества
- Универсальный класс для быстрых экспериментов с новыми моделями
- Минимизация рутины при обучении и валидации
- Чистая структура проекта, удобная для расширения
- Простое воспроизведение результатов

## 🔧 Требования
- Python 3.9+
- PyTorch 2.0+
- NumPy, Matplotlib, Pillow

### Установка зависимостей:

```bash
pip install -r requirements.txt
```