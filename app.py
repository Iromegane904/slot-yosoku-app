# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from train_model import cross_validate  # 交差検証関数を再利用
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
import os

# ファイルパス
MODEL_ATARI_PATH = 'model/stacked_model_atari.pkl'
MODEL_RENCHAN_PATH = 'model/stacked_model_renchan.pkl'
DATA_PATH = 'data/slot_takusan.csv'

# モデル読み込み
model_atari = joblib.load(MODEL_ATARI_PATH)
model_renchan = joblib.load(MODEL_RENCHAN_PATH)

st.title("アレーティア")

st.header("台データを入力してください")

# 入力フォーム
with st.form(key='predict_form'):
    総回転数 = st.number_input('総回転数', min_value=0, value=500)
    現在G = st.number_input('現在G', min_value=0, value=100)
    平均当選G = st.number_input('平均当選G', min_value=0, value=400)
    差枚 = st.number_input('差枚', min_value=-10000, max_value=10000, value=0)
    前回当選G = st.number_input('前回当選G', min_value=0, value=200)
    前回連チャン数 = st.number_input('前回連チャン数', min_value=0, value=2)
    三連以上連チャン数 = st.number_input('3連以上の連チャン数', min_value=0, value=1)
    駆け抜け = st.selectbox('駆け抜け（単発か？）', options=[0,1])
    当たり回数 = st.number_input('当たり回数', min_value=0, value=3)
    submit_button = st.form_submit_button('予測する')

if submit_button:
    input_data = pd.DataFrame({
        '総回転数': [総回転数],
        '現在G': [現在G],
        '平均当選G': [平均当選G],
        '差枚': [差枚],
        '前回当選G': [前回当選G],
        '前回連チャン数': [前回連チャン数],
        '3連以上の連チャン数': [三連以上連チャン数],
        '駆け抜け': [駆け抜け],
        '当たり回数': [当たり回数]
    })

    # 予測
    pred_atari = model_atari.predict(input_data)[0]
    pred_renchan = model_renchan.predict(input_data)[0]

    st.subheader("予測結果")
    st.write(f"次の当たりゲーム数（予測）: {int(pred_atari)}G")
    st.write(f"次の連チャン数（予測）: {round(pred_renchan, 1)}連")

    st.session_state['input_data'] = input_data  # セッションに保存

# 実際の結果入力
st.header("実際の結果を入力してください（任意）")

if 'input_data' in st.session_state:
    with st.form(key='result_form'):
        actual_atari = st.number_input('実際の当たりゲーム数', min_value=0, value=0)
        actual_renchan = st.number_input('実際の連チャン数', min_value=0, value=0)
        save_button = st.form_submit_button('結果を保存')

    if save_button:
        input_data = st.session_state['input_data']
        input_data['次の当たりゲーム数'] = actual_atari
        input_data['次の連チャン数'] = actual_renchan

        # CSVに追記
        if os.path.exists(DATA_PATH):
            df_existing = pd.read_csv(DATA_PATH)
            df_updated = pd.concat([df_existing, input_data], ignore_index=True)
        else:
            df_updated = input_data

        df_updated.to_csv(DATA_PATH, index=False)
        st.success("結果を保存しました。")

# モデル再学習
st.header("モデルを再学習")

if st.button('再学習する'):
    df = pd.read_csv(DATA_PATH)
    X = df[['総回転数', '現在G', '平均当選G', '差枚', '前回当選G', '前回連チャン数', '3連以上の連チャン数', '駆け抜け', '当たり回数']]
    y_atari = df['次の当たりゲーム数']
    y_renchan = df['次の連チャン数']

    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    xgb_model = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)

    stacked_model_atari = StackingRegressor(estimators=[
        ('rf', rf_model),
        ('xgb', xgb_model)
    ], final_estimator=LinearRegression())

    stacked_model_renchan = StackingRegressor(estimators=[
        ('rf', rf_model),
        ('xgb', xgb_model)
    ], final_estimator=LinearRegression())

    stacked_model_atari.fit(X, y_atari)
    stacked_model_renchan.fit(X, y_renchan)

    joblib.dump(stacked_model_atari, MODEL_ATARI_PATH)
    joblib.dump(stacked_model_renchan, MODEL_RENCHAN_PATH)

    st.success("再学習が完了しました。最新モデルに更新しました。")
