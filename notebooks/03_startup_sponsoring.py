#!/usr/bin/env python
# coding: utf-8

# # Риск и распределение инвестиций в стартапы: расчёт объёма инвестиций
# 
# ---
# 
# # Цель:
# - Написать модель машинного обучения для:
#     - рассчёта оптимального объёма инвестиций в стартап;
#     - выявление признаков, влияющих на объём инвестиций;
#     - сохранение модели для дальнейшего использования (расчёт объёма инвестиций).

# In[1]:


import os
import numpy as np
import warnings
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
plt.rcParams['figure.figsize'] = (8, 7)
get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv('../data/processed/startup_investment_dataset+rejected_status.csv')


# In[3]:


df.head()


# In[4]:


df = df[df['is_rejected'] == 0].drop(columns=['is_rejected', 'market_size_estimate'])
print(f'Общая длина: {len(df)} стартапов')
df.head()


# In[5]:


sns.boxplot(y=df['investment_amount'])
plt.show()


# In[6]:


# Избавление от выбросов
q_low = df['investment_amount'].quantile(0.01)
q_high = df['investment_amount'].quantile(0.95)

df = df[(df['investment_amount'] >= q_low) & (df['investment_amount'] <= q_high)]

print(f'Отобрано {len(df)} стартапов с объёмом инвестиций от ${int(q_low):,} до ${int(q_high):,}.')


# In[7]:


sns.histplot(data=df, x='investment_amount')
plt.title('Инвестиции в стартапы')
plt.grid(axis='y')
plt.show()


# In[8]:


sns.boxplot(y=df['investment_amount'])
plt.show()


# In[9]:


color_map = {
    'requested_amount': 'magenta',
    'team_size': 'red',
    'annual_revenue': 'green',
    'pre_money_valuation': 'purple',
    'founders_experience_years': 'orange'
}

for feature in df.select_dtypes(include=['int64', 'float64']).columns:
    if feature != 'investment_amount':
        sns.lmplot(
            data=df,
            x=feature,
            y='investment_amount',
            scatter_kws={'color': color_map.get(feature, 'gray')},
        )
        plt.grid(axis='y')
        plt.title(f'{feature} к объёму инвестиций')
        plt.show()


# # Моделирование

# In[10]:


import mlflow
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.cluster import KMeans

from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


# ## Feature engineering
# 
# Добавление новых признаков для лучшего обучения модели:
# 
# - Добавление отношений денежных признаков;
# - Добавление производных признаков от команды;
# - Бизнес-признаки;
# - Логарифмирование денежных и остальных числовых признаков;

# In[11]:


def create_additional_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    eps = 1

    # --- Отношения денежных признаков ---
    df['requested_to_valuation'] = (df['requested_amount'] / (df['pre_money_valuation'] + eps))
    df['revenue_to_valuation'] = (df['annual_revenue'] / (df['pre_money_valuation'] + eps))
    df['revenue_per_employee'] = (df['annual_revenue'] / (df['team_size'] + eps))

    # --- Команда ---
    df['experience_per_member'] = (df['founders_experience_years'] / (df['team_size'] + eps))
    df['team_maturity'] = (df['team_size'] * df['founders_experience_years'])

    # --- Бизнес-флаги ---
    df['early_stage'] = (df['startup_stage'].isin(['Idea', 'Pre-Seed']).astype(int))
    df['is_us_market'] = ((df['region'] == 'US').astype(int))

    # --- Логарифмы денежных признаков ---

    numeric_features = df.select_dtypes(include=['int64', 'float64'])

    for col in numeric_features:
        if col in df.columns:
            df[f'{col}_log'] = np.log1p(df[col])

    # Очистка проблем после деления
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    return df


# In[12]:


df = create_additional_features(df)
df.head()


# In[13]:


# Логарифмирование целевой переменной
df['investment_amount_log'] = np.log1p(df['investment_amount'])

target = 'investment_amount_log'


# # Кластеризация стартапов
# 
# - В распределении объёмов инвестиций наблюдается сильный скос вправо, что говорит о наличии отлично высоких объёмах, следовательно модель не сможет стабильно работать на всей выборке
# - Кластеризация стартапов поможет разбить их на группы, на каждую из которых можно создать уникальную регрессионную модель.
# - Идея кластеризации заключается в разделении стартапов на сегменты: __дешёвые__, __средние__ и ___дорогие__, при том __без использования целевой переменной__, поскольку кластер будет являться __прокси-признаком__, вызывающим:
#     - Утечку данных;
#     - Невозможность присвоить кластер при оценке новых стартапов.
# 
# В качестве признаков для сегментации используются:
# 
# - `requested_amount` — запрошенный объём инвестиций;
# - `pre_money_valuation` — предварительная оценка стартапа;
# - `annual_revenue` — существующая годовая выручка стартапа;
# - `team_size` — количество сотрудников/участников;
# - `founders_experience_years` — опыт сотрудников/участников.

# In[14]:


cluster_preprocessor = Pipeline(steps=[
    ('scaler', StandardScaler())
])

cluster_preprocessor


# In[15]:


cluster_features = [
    'requested_amount',
    'pre_money_valuation',
    'annual_revenue',
    'team_size',
]

kmeans = Pipeline(steps=[
    ('preprocessor', cluster_preprocessor),
    ('kmeans', KMeans(n_clusters=4, random_state=42))
])

kmeans.fit(df[cluster_features])

clusters = kmeans.predict(
    df[cluster_features]
)

df['cluster'] = clusters
df.head()


# In[16]:


display(
    df[
        df.cluster==2
    ][[
        'startup_stage',
        'requested_amount',
        'pre_money_valuation',
        'annual_revenue',
        'team_size',
        'investment_amount'
    ]].head(15)
)


# In[17]:


ax = (
    df.drop(columns=['investment_amount'])
        .corr(numeric_only=True)['cluster']
        .sort_values(ascending=False).plot(kind='bar', color='lime')
     )
plt.axhline(y=0, linestyle='--', c='red')
plt.title('Корреляция признаков к вычисленному присвоеному кластеру')
plt.show()

# ---------------------------------

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))

# Clusters palette
palette = {
    0: "royalblue",
    1: "limegreen",
    2: "red",       # обязательно красный
    3: "darkorange"
}

df_sorted = df.sort_values(by='investment_amount', ascending=True)

sns.histplot(data=df_sorted, x='investment_amount', hue='cluster', palette=palette, alpha=.7, ax=axes[0], bins=80)
axes[0].set_title('Объём инвестиций по кластерам')
axes[0].set_xlabel('Investment Amount')

sns.countplot(data=df, x='cluster', hue='cluster', palette=palette, ax=axes[1])
# Подписи над столбцами
for container in axes[1].containers:
    axes[1].bar_label(container)
axes[1].set_title('Количество стартапов в кластерах')
axes[1].set_xlabel('Cluster')
plt.tight_layout()

plt.show()


# In[18]:


### print('Процентное соотношение сегментов:\n', df['cluster'].value_counts(normalize=True).sort_index() * 100)

cluster_stats = df.groupby('cluster').mean(numeric_only=True).sort_values(by='requested_amount')
display(cluster_stats)


# ### > Анализ 2-го кластера

# In[19]:


print(df['cluster'].value_counts())
print(df['cluster'].value_counts(normalize=True)*100)


# In[20]:


df[df['cluster'] == 2].head(7)


# In[34]:


try:
    joblib.dump(kmeans, '../src/models/models_to_use/clusteriser.joblib')
    print('Модель кластеризации сохранена.')
except Exception as e:
    print(f'Не удалось сохранить препроцессор/модель кластеризации.\n{e}')


# ---
# 
# ## Кластеризация стартапов
# 
# Для дополнительной сегментации стартапов была проведена кластеризация методом K-Means. Целью являлось выделение групп компаний со схожими финансовыми и организационными характеристиками для дальнейшего использования в инвестиционном пайплайне.
# 
# Кластеризация позволяет разделить стартапы на группы со схожим профилем риска и масштаба, а затем использовать специализированные модели регрессии для каждого сегмента отдельно.
# 
# ### Использованные признаки
# 
# Для обучения использовались признаки, отражающие зрелость компании и масштаб бизнеса:
# 
# - `requested_amount`
# - `pre_money_valuation`
# - `annual_revenue`
# - `team_size`
# - `founders_experience_years`
# 
# При построении модели не использовались:
# 
# - `investment_amount`
# 
# Это исключает утечку данных (data leakage), поскольку будущий объём инвестиций не участвует в формировании сегментов.
# 
# ---
# 
# ## Распределение стартапов по кластерам
# 
# | Кластер | Количество | Доля |
# |----------|------------:|------:|
# | 3 | 783 | 42.8% |
# | 1 | 737 | 40.3% |
# | 0 | 283 | 15.5% |
# | 2 | 27 | 1.5% |
# 
# Основную часть выборки составляют два массовых сегмента: ранние и зрелые стартапы. Также были обнаружены более крупные компании и небольшой специализированный сегмент высокодоходных стартапов.
# 
# ---
# 
# ## Средние значения ключевых признаков
# 
# | Кластер | Requested Amount | Pre-money Valuation | Annual Revenue | Investment Amount |
# |----------|----------------:|--------------------:|----------------:|------------------:|
# | 3 | 1.33M | 7.03M | 0.17M | 1.12M |
# | 1 | 2.87M | 15.04M | 0.66M | 2.46M |
# | 0 | 7.88M | 44.94M | 0.72M | 6.32M |
# | 2 | 5.73M | 31.48M | 7.17M | 4.54M |
# 
# Наблюдается закономерное увеличение размеров компаний:
# 
# ```text
# Cluster 3 → Cluster 1 → Cluster 0
# 
# ранние → зрелые → крупные
# ```
# 
# Однако был обнаружен дополнительный сегмент:
# 
# ```text
# Cluster 2
# 
# высокодоходные scale-up компании
# ```
# 
# ---
# 
# ## Интерпретация кластеров
# 
# ### Cluster 3 — Early Stage Startups
# 
# Наиболее ранний сегмент.
# 
# Характеристики:
# 
# - высокая доля Idea и Pre-Seed
# - небольшие команды
# - низкий объём выручки
# - минимальные объёмы инвестиций
# - высокая неопределённость
# 
# Этот кластер соответствует типичным ранним стартапам, находящимся на стадии поиска продукта и рынка.
# 
# ---
# 
# ### Cluster 1 — Mature Startups
# 
# Основной сегмент выборки.
# 
# Характеристики:
# 
# - преобладают стадии Seed и Series A
# - сформированные команды
# - умеренные показатели оценки
# - средний инвестиционный объём
# 
# Представляет типичный венчурный рынок.
# 
# ---
# 
# ### Cluster 0 — Large Investment Startups
# 
# Крупные инвестиционные проекты.
# 
# Характеристики:
# 
# - высокие оценки компаний
# - крупные команды
# - значительные инвестиционные запросы
# - поздние стадии развития
# 
# Данный сегмент соответствует стартапам, требующим повышенного объёма капитала и более серьёзной оценки рисков.
# 
# ---
# 
# ### Cluster 2 — High-Revenue Scale-Up Startups
# 
# Наиболее редкий сегмент (~1.5%).
# 
# Особенность данного кластера — аномально высокий объём выручки:
# 
# - Annual Revenue ≈ 7.17M
# - преобладают стадии Series A / Series B
# - опытные команды
# - высокий уровень зрелости бизнеса
# 
# Несмотря на относительно небольшой размер сегмента, алгоритм K-Means выделил данные компании отдельно.
# 
# Вероятная интерпретация:
# 
# это не выбросы, а специализированный класс зрелых компаний, уже имеющих существенную выручку и находящихся на стадии масштабирования бизнеса (scale-up).
# 
# ---
# 
# ## Вывод
# 
# Кластеризация выявила устойчивую структуру рынка стартапов и выделила четыре различных сегмента компаний:
# 
# - ранние стартапы
# - зрелые компании
# - крупные инвестиционные проекты
# - высокодоходные scale-up стартапы
# 
# Полученная сегментация может использоваться как дополнительный этап инвестиционного пайплайна:
# 
# ```text
# Startup Features
#         ↓
# Cluster Prediction
#         ↓
# Cluster-specific Regressor
#         ↓
# Investment Amount Prediction
# ```
# 
# Такой подход позволяет строить специализированные модели под разные типы компаний и потенциально снижать ошибку прогнозирования инвестиционного объёма.

# # Регрессия
# 
# Распределение объёма инвестиций имеет выраженную положительную асимметрию: наблюдается длинный хвост крупных сделок. Это создаёт проблему дисбаланса — модели начинают сильнее оптимизироваться под большие инвестиции, ухудшая качество прогнозирования для основной части выборки.
# 
# Для снижения влияния перекоса применялось логарифмирование целевой переменной:
# 
# ```python
# investment_amount_log = np.log1p(investment_amount)
# ```
# 
# В качестве функции оптимизации используется **корень из среднеквадратичной ошибки (RMSE)**, поскольку данная метрика чувствительна к большим отклонениям и позволяет сильнее штрафовать крупные ошибки.
# 
# Основной метрикой оценки качества выбрана **MAPE (Mean Absolute Percentage Error)**, поскольку она показывает среднюю относительную ошибку прогноза и лучше интерпретируется с бизнес-точки зрения.
# 
# Дополнительно анализируются:
# 
# - RMSE
# - MAE
# - R^2
# 
# Учитывая:
# 
# - высокий уровень шума;
# - табличный характер данных;
# - наличие нелинейных зависимостей;
# - ограниченный объём выборки;
# 
# для валидации были выбраны следующие алгоритмы:
# 
# - RandomForestRegressor
# - GradientBoostingRegressor
# - XGBoostRegressor
# - LightGBMRegressor
# 
# Линейные модели и нейросетевые архитектуры не были выбраны как основные кандидаты, поскольку предварительные эксперименты показали их меньшую устойчивость на данной задаче.

# ## Разбиение и преобразование данных

# In[33]:


num_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
remove_cols = ['investment_amount', 'investment_amount_log', 'cluster']

num_features = [col for col in num_features if col not in remove_cols]

print(f'Числовые признаки: {num_features}')
print(f'Категориальные признаки: {cat_features}')


# In[35]:


target='investment_amount_log'

X=df.drop(columns=['investment_amount', 'investment_amount_log'])

y=df[target]


# In[36]:


print(X.columns)


# In[37]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42)


# # Обучение моделей машинного обучения

# In[43]:


MODELS = {
    'random_forest': (
        RandomForestRegressor(
            random_state=42,
            n_jobs=-1
        ),
        {
            'model__n_estimators': [200, 500],
            'model__max_depth': [None, 10, 20]
        }
    ),

    'gboost': (
        GradientBoostingRegressor(
            random_state=42
        ),
        {
            'model__n_estimators': [200, 500],
            'model__learning_rate': [0.05, 0.1],
            'model__max_depth': [3, 5]
        }
    ),

    'xgb': (
        XGBRegressor(
            random_state=42,
            n_jobs=-1,
            verbosity=0
        ),
        {
            'model__n_estimators': [200, 500],
            'model__learning_rate': [0.01, 0.05, 0.1],
            'model__max_depth': [3, 5, 7],
            'model__min_child_weight': [1, 5],
            'model__subsample': [0.7, 0.9],
            'model__colsample_bytree': [0.7, 0.9],
            'model__reg_alpha': [0, 0.1, 1],
            'model__reg_lambda': [1, 5]
        }
    ),

    'lgbm': (
        LGBMRegressor(
            random_state=42,
            n_jobs=-1,
            verbose=-1
        ),
        {
            'model__n_estimators': [200, 500],
            'model__learning_rate': [0.01, 0.05, 0.1],
            'model__max_depth': [-1, 5, 10],
            'model__num_leaves': [31, 63],
            'model__min_child_samples': [20, 50],
            'model__subsample': [0.7, 0.9],
            'model__colsample_bytree': [0.7, 0.9]
        }
    )
}

MODELS.keys()


# In[48]:


trained_models={}


# In[51]:


def train_models(
    X_train,
    X_test,
    y_train,
    y_test,
    cluster_id,
    report_dir="artifacts/reports/regression",
    model_dir="artifacts/models"
):

    cluster_num_features=X_train.select_dtypes(include=['int64','float64']).columns.tolist()
    cluster_cat_features=X_train.select_dtypes(include=['object']).columns.tolist()

    cluster_preprocessor=ColumnTransformer([
        ('scaler', StandardScaler(), cluster_num_features),
        ('encoder', OneHotEncoder(handle_unknown='ignore'), cluster_cat_features)
    ])

    os.makedirs(report_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    results=[]

    for model_name,(model,param_grid) in MODELS.items():
        print("\n"+"="*60)
        print(f"CLUSTER {cluster_id} | MODEL {model_name.upper()}")
        print("="*60)

        pipeline=Pipeline([
            ('preprocessor', cluster_preprocessor),
            ('model', model)
        ])

        with mlflow.start_run(
            run_name=f'cluster_{cluster_id}_{model_name}'):
            mlflow.set_tag("task", "regression")
            mlflow.set_tag("cluster", cluster_id)
            mlflow.set_tag("model_name", model_name)

            print("GridSearchCV running...")

            grid=GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                scoring='neg_root_mean_squared_error', 
                cv=5,
                n_jobs=-1
            )

            grid.fit(X_train,y_train)
            best_model=(grid.best_estimator_)

            # predictions
            y_pred_log=(best_model.predict(X_test))
            y_pred=np.expm1(y_pred_log)
            y_true=np.expm1(y_test)

            rmse=np.sqrt(mean_squared_error(y_true,y_pred))
            mae=mean_absolute_error(y_true,y_pred)
            r2=r2_score(y_true, y_pred)
            mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true,1)))

            print(f'RMSE:{rmse:,.0f}')
            print(f'MAE:{mae:,.0f}')
            print(f'R2:{r2:.4f}')
            print(f'MAPE:{mape:.2%}')

            # save model
            model_path=(f"{model_dir}/cluster_{cluster_id}_{model_name}.joblib")

            joblib.dump(best_model, model_path)

            trained_models[f'cluster_{cluster_id}_{model_name}']=best_model

            # mlflow
            mlflow.log_params(grid.best_params_)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)
            mlflow.log_metric("mape", mape)
            mlflow.log_metric("cv_rmse", abs(grid.best_score_))


            mlflow.sklearn.log_model(best_model, "model")

            # report
            report_path=os.path.join(report_dir, f'cluster_{cluster_id}_{model_name}_report.txt')

            with open(report_path, "w", encoding="utf-8") as f:
                f.write(f"CLUSTER: {cluster_id}\n")
                f.write(f"MODEL {model_name}\n\n")
                f.write(f"RMSE: {rmse:.2f}\n")
                f.write(f"MAE: {mae:.2f}\n")
                f.write(f"R2: {r2:.4f}\n")
                f.write(f"MAPE: {mape:.2%}\n\n")
                f.write("BEST PARAMS:\n")

                for k,v in (grid.best_params_.items()):

                    f.write(f"{k}: {v}\n")

            mlflow.log_artifact(report_path)

            results.append({
                'cluster': cluster_id,
                'model': model_name,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'mape': mape})


    return (pd.DataFrame(results).sort_values(by='mape'))


# In[52]:


for cluster_id in sorted(X_train['cluster'].unique()):
    train_mask=(X_train['cluster']==cluster_id)
    test_mask=(X_test['cluster']==cluster_id)

    X_cluster_train=(X_train[train_mask].drop(columns='cluster'))
    X_cluster_test=(X_test[test_mask].drop(columns='cluster'))
    y_cluster_train=(y_train.loc[X_cluster_train.index])
    y_cluster_test=(y_test.loc[X_cluster_test.index])

    result = train_models(X_cluster_train, X_cluster_test, y_cluster_train, y_cluster_test, cluster_id)

    cluster_results.append(result)


# In[54]:


trained_models


# In[ ]:




