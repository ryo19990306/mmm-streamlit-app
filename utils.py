
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

def run_forecast(budget, start_date, end_date):
    # ダミー処理（実際はMMMロジックをここに入れる）
    dates = pd.date_range(start=start_date, end=end_date)
    sales = np.random.randint(1000, 2000, size=len(dates)) * (budget / 10000)
    df = pd.DataFrame({"date": dates, "Predicted_Sales": sales})

    # グラフ保存
    image_path = "sales_prediction_plot.png"
    plt.figure(figsize=(10, 4))
    plt.plot(df["date"], df["Predicted_Sales"], label="Predicted_Sales")
    plt.title("将来予測 売上")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.tight_layout()
    plt.savefig(image_path)
    plt.close()

    # 評価指標（ダミー）
    evaluation_df = pd.DataFrame({
        "指標": ["R_squared", "MAPE", "RMSE"],
        "値": [0.89, 12.5, 340.2]
    })

    return df, evaluation_df, image_path
