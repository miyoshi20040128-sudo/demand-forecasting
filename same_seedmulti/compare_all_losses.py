import os
import tomllib

import pandas as pd
import matplotlib.pyplot as plt


def create_combined_report(config=None):
    if config is None:
        with open("config.toml", "rb") as f:
            config = tomllib.load(f)

    output_dir = config["path"]["output_dir"]

    loss_names = config.get("compare", {}).get(
        "loss_names",
        ["mse", "mae", "huber", "dilate", "derivative"],
    )

    metric_rows = []
    prediction_data = {}

    for loss_name in loss_names:
        metric_path = os.path.join(
            output_dir,
            loss_name,
            f"metrics_recorded_sales_{loss_name}.csv",
        )

        pred_path = os.path.join(
            output_dir,
            loss_name,
            f"prediction_result_{loss_name}.csv",
        )

        if not os.path.exists(metric_path):
            print(f"Skip metrics: {metric_path} が見つかりません")
            continue

        if not os.path.exists(pred_path):
            print(f"Skip prediction: {pred_path} が見つかりません")
            continue

        metric_df = pd.read_csv(metric_path)
        pred_df = pd.read_csv(pred_path)

        pred_df["date"] = pd.to_datetime(pred_df["date"])
        pred_df = pred_df.sort_values("date").reset_index(drop=True)

        row = {
            "loss": loss_name,
            "best_valid_loss": metric_df.loc[0, "best_valid_loss"],
            "MAE": metric_df.loc[0, "mae_recorded_sales"],
            "RMSE": metric_df.loc[0, "rmse_recorded_sales"],
            "MAPE": metric_df.loc[0, "mape_recorded_sales"],
        }

        metric_rows.append(row)
        prediction_data[loss_name] = pred_df

    if len(metric_rows) == 0:
        raise FileNotFoundError(
            "比較用の metrics / prediction_result が見つかりませんでした。"
        )

    compare_df = pd.DataFrame(metric_rows)

    # MAPEが小さい順に並べる
    compare_df = compare_df.sort_values("MAPE").reset_index(drop=True)

    display_df = compare_df.copy()
    display_df["best_valid_loss"] = display_df["best_valid_loss"].round(5)
    display_df["MAE"] = display_df["MAE"].round(3)
    display_df["RMSE"] = display_df["RMSE"].round(3)
    display_df["MAPE"] = display_df["MAPE"].round(2)

    print("========== Loss Comparison ==========")
    print(display_df.to_string(index=False))
    print("=====================================")

    csv_path = os.path.join(output_dir, "loss_comparison.csv")
    display_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # 5個の損失関数を想定
    # 1行目: 表
    # 2〜4行目: グラフ (3x2のうち5個使う)
    fig = plt.figure(figsize=(20, 20))
    gs = fig.add_gridspec(4, 2, height_ratios=[1.2, 1.6, 1.6, 1.6])

    # =========================
    # 表
    # =========================
    ax_table = fig.add_subplot(gs[0, :])
    ax_table.axis("off")

    table = ax_table.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        cellLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.8)

    ax_table.set_title(
        "Loss Function Comparison",
        fontsize=18,
        pad=20,
    )

    # =========================
    # グラフ
    # =========================
    plot_order = loss_names

    for i, loss_name in enumerate(plot_order):
        row_idx = 1 + i // 2
        col_idx = i % 2

        ax = fig.add_subplot(gs[row_idx, col_idx])

        if loss_name not in prediction_data:
            ax.axis("off")
            ax.set_title(f"{loss_name} (no data)")
            continue

        pred_df = prediction_data[loss_name].copy()
        pred_df = pred_df.sort_values("date").reset_index(drop=True)

        ax.plot(pred_df["date"], pred_df["actual"], label="Actual")
        ax.plot(pred_df["date"], pred_df["prediction"], label="Prediction")

        ax.set_title(loss_name)
        ax.set_xlabel("date")
        ax.set_ylabel("recorded_sales")
        ax.grid(True)
        ax.tick_params(axis="x", rotation=45)

        metric_row = display_df[display_df["loss"] == loss_name]
        if len(metric_row) > 0:
            mae = metric_row.iloc[0]["MAE"]
            rmse = metric_row.iloc[0]["RMSE"]
            mape = metric_row.iloc[0]["MAPE"]

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

    # 空き枠を消す
    total_axes = 6
    used_axes = len(plot_order)
    for j in range(used_axes, total_axes):
        row_idx = 1 + j // 2
        col_idx = j % 2
        ax_empty = fig.add_subplot(gs[row_idx, col_idx])
        ax_empty.axis("off")

    plt.tight_layout()

    jpg_path = os.path.join(output_dir, "loss_comparison_with_graphs.jpg")
    plt.savefig(jpg_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved comparison CSV to: {csv_path}")
    print(f"Saved comparison JPG to: {jpg_path}")


def main():
    with open("config.toml", "rb") as f:
        config = tomllib.load(f)

    create_combined_report(config)


if __name__ == "__main__":
    main()