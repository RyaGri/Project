import os
import pandas as pd
from typing import List
from datetime import datetime
from catboost import CatBoostClassifier
from fastapi import Depends, FastAPI
from schema import PostGet, Response
from loguru import logger
from sqlalchemy import create_engine
import hashlib

app = FastAPI()

def get_model_path(path: str) -> str:
    if os.environ.get("IS_LMS") == "1":  # проверяем где выполняется код в лмс, или локально. Немного магии
        MODEL_PATH = '/workdir/user_input/' + path
    else:
        MODEL_PATH = path
    return MODEL_PATH


SALT = "your_salt_value"
GROUPS = ["control", "test"]

def get_exp_group(user_id: int, salt: str = SALT, groups: list = GROUPS) -> str:
    value_str = str(user_id) + salt
    hash_value = int(hashlib.sha256(value_str.encode()).hexdigest(), 16)
    num_groups = len(groups)
    group_index = hash_value % num_groups
    group = groups[group_index]
    return group

# загружаем модель
def load_model(type_model: str):    
    if type_model == "control":
        model_path = get_model_path("model_control")
    elif type_model == "test":
        model_path = get_model_path("model_test")
    else:
        raise ValueError('unknown group') 
    model = CatBoostClassifier()
    model.load_model(model_path)
    return model

# функция для загрузки больших масивов частями
def batch_load_sql(query: str) -> pd.DataFrame:
    CHUNKSIZE = 200000
    engine = create_engine(
        "postgresql://robot-startml-ro:pheiph0hahj1Vaif@"
        "postgres.lab.karpov.courses:6432/startml"
    )
    conn = engine.connect().execution_options(stream_results=True)
    chunks = []
    for chunk_dataframe in pd.read_sql(query, conn, chunksize=CHUNKSIZE):
        chunks.append(chunk_dataframe)
    conn.close()
    return pd.concat(chunks, ignore_index=True)

# загружаем признаки
def load_features_post_model_test() -> pd.DataFrame:
    df_post_data = batch_load_sql("""SELECT * FROM ryabgri_post_inf_features_emb_3""")
    df_post_data = df_post_data.drop(['index'], axis=1)
    return df_post_data

def load_features_post_model_control() -> pd.DataFrame:
    df_post_data = batch_load_sql("""SELECT * FROM ryagrig_posts_info_lesson_22""")
    df_post_data = df_post_data.drop(['index'], axis=1)
    return df_post_data

def load_features_user() -> pd.DataFrame:
    df_user_data = batch_load_sql("""SELECT * FROM ryagrig_users_lesson_22""")
    df_user_data = df_user_data.drop(['index'], axis=1)
    return df_user_data


def load_liked_posts() -> pd.DataFrame:
    liked_posts_query = """
        SELECT distinct post_id, user_id
        FROM public.feed_data
        WHERE action='like'
        """
    liked_posts = batch_load_sql(liked_posts_query)
    return liked_posts


logger.info("loading control model")
model_control = load_model("control")

logger.info("loading test model")
model_test = load_model("test")

logger.info("loading user features")
df_user_data = load_features_user()

logger.info("loading post control features")
df_post_data_control = load_features_post_model_control()

logger.info("loading post test features")
df_post_data_test = load_features_post_model_test()

logger.info("loading liked posts")
liked_posts = load_liked_posts()

logger.info("service up")


@app.get("/post/recommendations/", response_model=Response)
def recommended_posts(id: int, time: datetime, limit: int = 10) -> Response:
    return get_recommended_feed(id, time, limit)
    
def get_recommended_feed(
        id: int, 
        time: datetime, 
        limit: int,
        df_post_data_control: pd.DataFrame = df_post_data_control,
        df_post_data_test: pd.DataFrame = df_post_data_test,
        liked_posts: pd.DataFrame = liked_posts,
        model_control=model_control,
        model_test=model_test):  
    
    num_of_posts = 100 # Чтобы не нагружать инфраструктуру сделаем ограничение по кол-ву постов в predict 
    
    group = get_exp_group(id)
    logger.info(f"group: {group}")
    df_post_data = df_post_data_control if group == "control" else df_post_data_test
    model = model_control if group == "control" else model_test
    
    df_user_id = df_user_data.loc[df_user_data.user_id == id]  # Получение данных о конкретном пользователе по его id
    df_user_id = df_user_id.drop('user_id', axis=1)  # Удаление столбца 'user_id'
    logger.info("5")
    
    # Определяем категории постов, где пользователь поставил наибольшее число лайков
    user_topic = df_user_id[['business', 'covid', 'entertainment', 'movie', 'politics', 'sport', 'tech']]  # Извлечение категорий постов, где пользователь поставил наибольшее число лайков
    user_topic = user_topic.iloc[0]  # Получение первой строки (данные пользователя)
    user_topic = user_topic.sort_values(ascending=False).iloc[:3].index.to_list()  # Сортировка категорий по убыванию и выбор первых трех
    logger.info("6")

    liked_posts_ = liked_posts[liked_posts.user_id == id].post_id.values  # Идентификаторы постов, лайкнутых пользователем
    
    post_pred = [] # Определим посты
    for utopic in user_topic:
        post_pred_ = df_post_data[(df_post_data['topic'] == utopic) & (~df_post_data['post_id'].isin(liked_posts_))]['post_id'].iloc[:num_of_posts].values   # Получение  идентификаторов постов для каждой категории (исключая лайкнутые)
        post_pred.extend(post_pred_)  # Добавление идентификаторов в список
        
    logger.info("7")
    
    # посты, по которым будет вычислено предсказание
    post_candidate = df_post_data[df_post_data.post_id.isin(post_pred)]  # DataFrame с данными о постах, для которых будет выполнено предсказание
    posts_features = post_candidate.drop(['text'], axis=1)  # Отбрасывание столбца 'text'
    
    content = post_candidate[['post_id', 'text', 'topic']]  # DataFrame с данными о постах и их содержимом
    
    add_user_features = dict(zip(df_user_id.columns, df_user_id.values[0]))  # Добавление данных о пользователе в виде словаря
    
    logger.info("8")
    
    user_posts_features = posts_features.assign(**add_user_features)  # Добавление данных о пользователе в DataFrame с данными о постах
    user_posts_features = user_posts_features.set_index('post_id')  # Изменение индекса на столбец 'post_id'
    
    if group == "test":
        user_posts_features['hour'] = time.hour  # Добавление столбца 'hour' с часом времени
        user_posts_features = user_posts_features.drop(['business', 'covid', 'entertainment', 'politics', 'movie', 'sport', 'tech'], axis = 1)  # Удаление категорий постов
    else:
        user_posts_features['dayofweek'] = time.weekday()
        user_posts_features['hour'] = time.hour
    
    predicts = model.predict_proba(user_posts_features)[:, 1]  # Предсказание рекомендаций на основе модели
    user_posts_features["predicts"] = predicts  # Добавление предсказаний в DataFrame с данными о постах
    
    liked_posts_ = liked_posts[liked_posts.user_id == id].post_id.values  # Идентификаторы лайкнутых постов
    
    filtered_ = user_posts_features[~user_posts_features.index.isin(liked_posts_)]  # Отфильтрованные посты (исключая лайкнутые)
    filtered_ = filtered_.sort_values('predicts')[-limit:]  # Выбор ограниченного количества рекомендуемых постов
    recomended_posts = filtered_.index  # Идентификаторы рекомендуемых постов
    logger.info("9")
    return Response(
        exp_group=group,
        recommendations=[
                         PostGet(**{"id": i,
                         "text": content[content.post_id == i].text.values[0],
                         "topic": content[content.post_id == i].topic.values[0]})
                          for i in recomended_posts
                        ]
           ) # Возвращение списка объектов PostGet с информацией о рекомендуемых постах, группу "control" или "test

logger.info("10")
