import os
import tomllib

import pandas as pd
import matplotlib.pyplot as plt


def main():
    with open("config.toml", "rb") as f:
        config = tomllib.load(f)

    output_dir = config["path"]["output_dir"]

    loss_names = [
        "mse",
        "mae",
        "huber",
        "dilate",
        "derivative",
    ]

    prediction_data = {}
    metric_data = {}

    for loss_name in loss_names:
        pred_path = os.path.join(
            output_dir,
            loss_name,
            f"prediction_result_{loss_name}.csv",
        )

        metric_path = os.path.join(
            output_dir,
            loss_name,
            f"metrics_recorded_sales_{loss_name}.csv",
        )

        if not os.path.exists(pred_path):
            print(f"Skip: {pred_path} が見つかりません")
            continue

        pred_df = pd.read_csv(pred_path)
        pred_df["date"] = pd.to_datetime(pred_df["date"])
        pred_df = pred_df.sort_values("date").reset_index(drop=True)
        prediction_data[loss_name] = pred_df

        if os.path.exists(metric_path):
            metric_df = pd.read_csv(metric_path)
            metric_data[loss_name] = {
                "MAE": round(metric_df.loc[0, "mae_recorded_sales"], 3),
                "RMSE": round(metric_df.loc[0, "rmse_recorded_sales"], 3),
                "MAPE": round(metric_df.loc[0, "mape_recorded_sales"], 2),
            }

    if len(prediction_data) == 0:
        raise FileNotFoundError(
            "prediction_result ファイルが見つかりませんでした。"
            "先に各損失関数で python main.py を実行してください。"
        )

    fig, axes = plt.subplots(3, 2, figsize=(18, 16))
    axes = axes.flatten()

    plot_order = [
        "mse",
        "mae",
        "huber",
        "dilate",
        "derivative",
    ]

    for i, loss_name in enumerate(plot_order):
        ax = axes[i]

        if loss_name not in prediction_data:
            ax.axis("off")
            ax.set_title(f"{loss_name} (no data)")
            continue

        df = prediction_data[loss_name]

        ax.plot(df["date"], df["actual"], label="Actual")
        ax.plot(df["date"], df["prediction"], label="Prediction")

        ax.set_title(f"{loss_name}")
        ax.set_xlabel("date")
        ax.set_ylabel("recorded_sales")
        ax.grid(True)
        ax.tick_params(axis="x", rotation=45)

        if loss_name in metric_data:
            mae = metric_data[loss_name]["MAE"]
            rmse = metric_data[loss_name]["RMSE"]
            mape = metric_data[loss_name]["MAPE"]

            metric_text = (
                f"MAE  = {mae}\n"
                f"RMSE = {rmse}\n"
                f"MAPE = {mape}%"
            )

            ax.text(
                0.02,
                0.95,
                metric_text,
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="top",
                bbox=dict(
                    boxstyle="round",
                    facecolor="white",
                    alpha=0.8,
                ),
            )

        ax.legend()

    # 6枠目は使わない
    for j in range(len(plot_order), len(axes)):
        axes[j].axis("off")

    plt.suptitle("Time Series Forecast Comparison by Loss Function", fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    save_path = os.path.join(output_dir, "prediction_graphs_comparison.jpg")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved comparison graph to: {save_path}")


if __name__ == "__main__":
    main()