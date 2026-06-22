import os
import tomllib

import numpy as np
import pandas as pd


def mean_absolute_percentage_error(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    nonzero = y_true != 0

    if nonzero.sum() == 0:
        return np.nan

    mape = np.mean(
        np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])
    ) * 100

    return mape


def main():
    with open("config.toml", "rb") as f:
        config = tomllib.load(f)

    sales_path = config["path"]["sales_path"]
    output_dir = config["path"]["output_dir"]

    area = config["data"]["area"]
    valid_item_index = str(config["data"]["valid_item_index"])

    prediction_path = os.path.join(output_dir, "prediction_result.csv")

    if not os.path.exists(prediction_path):
        raise FileNotFoundError(
            f"{prediction_path} が見つかりません。先に python main.py を実行してください。"
        )

    sales_df = pd.read_csv(sales_path)
    pred_df = pd.read_csv(prediction_path)

    sales_df["date"] = pd.to_datetime(sales_df["date"])
    pred_df["date"] = pd.to_datetime(pred_df["date"])

    sales_df["item_index"] = sales_df["item_index"].astype(str)

    # 評価対象の area / item に絞る
    sales_df = sales_df[
        (sales_df["area"] == area)
        & (sales_df["item_index"] == valid_item_index)
    ].copy()

    if "true_sales" not in sales_df.columns:
        raise ValueError("sales.csv に true_sales 列がありません。")

    if "prediction" not in pred_df.columns:
        raise ValueError("prediction_result.csv に prediction 列がありません。")

    # 念のため、予測結果に同じ日付が複数ある場合は平均する
    # 非重複7日予測なら基本的に重複しないはず
    pred_df = (
        pred_df
        .groupby("date", as_index=False)
        .agg({
            "prediction": "mean",
            "actual": "mean" if "actual" in pred_df.columns else "first",
        })
    )

    eval_df = pd.merge(
        pred_df,
        sales_df[["date", "true_sales", "recorded_sales"]],
        on="date",
        how="left",
    )

    eval_df = eval_df.dropna(subset=["true_sales", "prediction"]).reset_index(drop=True)

    if len(eval_df) == 0:
        raise ValueError(
            "prediction_result.csv と sales.csv の date が一致しませんでした。"
            "date形式や評価対象のitem_indexを確認してください。"
        )

    y_true = eval_df["true_sales"].astype(float)
    y_pred = eval_df["prediction"].astype(float)

    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mape = mean_absolute_percentage_error(y_true, y_pred)

    eval_df["error"] = eval_df["prediction"] - eval_df["true_sales"]
    eval_df["absolute_error"] = np.abs(eval_df["error"])

    nonzero = eval_df["true_sales"] != 0
    eval_df["absolute_percentage_error"] = np.nan
    eval_df.loc[nonzero, "absolute_percentage_error"] = (
        eval_df.loc[nonzero, "absolute_error"]
        / eval_df.loc[nonzero, "true_sales"]
    ) * 100

    print("===== Evaluation against true_sales =====")
    print(f"area: {area}")
    print(f"item_index: {valid_item_index}")
    print(f"evaluated days: {len(eval_df)}")
    print(f"date range: {eval_df['date'].min()} ~ {eval_df['date'].max()}")
    print("-----------------------------------------")
    print(f"MAE : {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"MAPE: {mape:.2f}%")
    print("=========================================")

    output_path = os.path.join(output_dir, "evaluation_true_sales.csv")
    eval_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Saved evaluation result to: {output_path}")


if __name__ == "__main__":
    main()