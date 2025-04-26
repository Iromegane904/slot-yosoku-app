# train_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# データ読み込み
df = pd.read_csv('slot_takusan.csv')

# 特徴量とターゲットに分割
X = df[['総回転数', '現在G', '平均当選G', '差枚', '前回当選G', '前回連チャン数', '3連以上の連チャン数', '駆け抜け', '当たり回数']]
y_atari = df['次の当たりゲーム数']
y_renchan = df['次の連チャン数']

# モデル作成
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

# 交差検証
kf = KFold(n_splits=5, shuffle=True, random_state=42)

def cross_validate(model, X, y):
    rmses = []
    r2s = []
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        r2 = r2_score(y_val, y_pred)
        rmses.append(rmse)
        r2s.append(r2)
    print(f"平均RMSE: {np.mean(rmses):.2f}")
    print(f"平均R2 : {np.mean(r2s):.2f}")

print("【次の当たりゲーム数モデルの交差検証】")
cross_validate(stacked_model_atari, X, y_atari)

print("\n【次の連チャン数モデルの交差検証】")
cross_validate(stacked_model_renchan, X, y_renchan)

# 最終学習（全データで学習して保存）
stacked_model_atari.fit(X, y_atari)
stacked_model_renchan.fit(X, y_renchan)

joblib.dump(stacked_model_atari, 'model/stacked_model_atari.pkl')
joblib.dump(stacked_model_renchan, 'model/stacked_model_renchan.pkl')

print("\nモデルを保存しました。")
