import os
import random

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from src.dataset import create_datasets
from src.model import ITransformer
from src.losses import get_loss_function


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def inverse_target_transform(values, scaler, feature_cols, target_idx):
    values = np.array(values).reshape(-1)

    dummy = np.zeros((len(values), len(feature_cols)))
    dummy[:, target_idx] = values

    inversed = scaler.inverse_transform(dummy)

    return inversed[:, target_idx]


def train(config):
    loss_name = config["train"].get("loss", "mse").lower()

    base_output_dir = config["path"]["output_dir"]
    output_dir = os.path.join(base_output_dir, loss_name)
    os.makedirs(output_dir, exist_ok=True)

    seed = config["train"].get("seed", 42)
    set_seed(seed)

    train_dataset, valid_dataset, info = create_datasets(config)

    batch_size = config["train"]["batch_size"]
    epochs = config["train"]["epochs"]
    learning_rate = config["train"]["learning_rate"]
    patience = config["train"]["patience"]

    seq_len = config["model"]["seq_len"]
    pred_len = config["model"]["pred_len"]
    d_model = config["model"]["d_model"]
    n_heads = config["model"]["n_heads"]
    num_layers = config["model"]["num_layers"]
    dropout = config["model"]["dropout"]

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    device = torch.device("cpu")

    model = ITransformer(
        seq_len=seq_len,
        pred_len=pred_len,
        num_features=len(info["feature_cols"]),
        d_model=d_model,
        n_heads=n_heads,
        num_layers=num_layers,
        dropout=dropout,
        target_idx=info["target_idx"],
    ).to(device)

    criterion = get_loss_function(loss_name, config=config)

    print("===== train setting =====")
    print(f"loss function: {loss_name}")
    print(f"output_dir: {output_dir}")
    print(f"device: {device}")
    print("=========================")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float("inf")
    wait = 0

    best_model_path = os.path.join(output_dir, f"best_itransformer_{loss_name}.pt")

    train_loss_history = []
    valid_loss_history = []

    for epoch in range(epochs):
        # =========================
        # train
        # =========================
        model.train()
        train_losses = []

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            preds = model(X_batch)
            loss = criterion(preds, y_batch)

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        # =========================
        # validation
        # =========================
        model.eval()
        valid_losses = []

        # DILATE loss は内部で torch.autograd.grad を使うので、
        # validation loss 計算時も torch.no_grad() を使わない
        if loss_name == "dilate":
            for X_batch, y_batch in valid_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                preds = model(X_batch)
                loss = criterion(preds, y_batch)

                valid_losses.append(loss.item())
        else:
            with torch.no_grad():
                for X_batch, y_batch in valid_loader:
                    X_batch = X_batch.to(device)
                    y_batch = y_batch.to(device)

                    preds = model(X_batch)
                    loss = criterion(preds, y_batch)

                    valid_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        valid_loss = np.mean(valid_losses)

        train_loss_history.append(train_loss)
        valid_loss_history.append(valid_loss)

        print(
            f"Epoch {epoch + 1:03d} | "
            f"train_loss: {train_loss:.5f} | "
            f"valid_loss: {valid_loss:.5f}"
        )

        # =========================
        # early stopping
        # =========================
        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            wait = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            wait += 1

        if wait >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    print(f"Best valid_loss: {best_val_loss:.5f}")
    print(f"Saved model to: {best_model_path}")

    # =========================
    # save loss history
    # =========================
    loss_df = pd.DataFrame({
        "epoch": np.arange(1, len(train_loss_history) + 1),
        "train_loss": train_loss_history,
        "valid_loss": valid_loss_history,
    })

    loss_csv_path = os.path.join(output_dir, f"loss_history_{loss_name}.csv")
    loss_df.to_csv(loss_csv_path, index=False, encoding="utf-8-sig")

    plt.figure(figsize=(10, 5))
    plt.plot(loss_df["epoch"], loss_df["train_loss"], label="train_loss")
    plt.plot(loss_df["epoch"], loss_df["valid_loss"], label="valid_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(f"Train Loss vs Validation Loss ({loss_name})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    loss_graph_path = os.path.join(output_dir, f"loss_curve_{loss_name}.png")
    plt.savefig(loss_graph_path)
    plt.close()

    # =========================
    # prediction
    # =========================
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    all_preds = []
    all_targets = []

    # ここでは loss を計算しないので no_grad でOK
    with torch.no_grad():
        for X_batch, y_batch in valid_loader:
            X_batch = X_batch.to(device)

            preds = model(X_batch)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(y_batch.numpy())

    preds_scaled = np.concatenate(all_preds, axis=0).reshape(-1)
    targets_scaled = np.concatenate(all_targets, axis=0).reshape(-1)

    predictions = inverse_target_transform(
        preds_scaled,
        scaler=info["scaler"],
        feature_cols=info["feature_cols"],
        target_idx=info["target_idx"],
    )

    actuals = inverse_target_transform(
        targets_scaled,
        scaler=info["scaler"],
        feature_cols=info["feature_cols"],
        target_idx=info["target_idx"],
    )

    result_df = pd.DataFrame({
        "window_id": info["val_window_ids"],
        "horizon": info["val_horizons"],
        "date": info["val_dates"],
        "actual": actuals,
        "prediction": predictions,
    })

    result_df = result_df.sort_values(
        ["date", "window_id", "horizon"]
    ).reset_index(drop=True)

    # =========================
    # evaluation against recorded_sales
    # =========================
    mae = np.mean(np.abs(result_df["actual"] - result_df["prediction"]))
    rmse = np.sqrt(np.mean((result_df["actual"] - result_df["prediction"]) ** 2))

    nonzero = result_df["actual"] != 0

    if nonzero.sum() > 0:
        mape = np.mean(
            np.abs(
                (result_df.loc[nonzero, "actual"] - result_df.loc[nonzero, "prediction"])
                / result_df.loc[nonzero, "actual"]
            )
        ) * 100
    else:
        mape = np.nan

    print("========== Evaluation against recorded_sales ==========")
    print(f"MAE : {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"MAPE: {mape:.2f}%")
    print("=======================================================")

    result_path = os.path.join(output_dir, f"prediction_result_{loss_name}.csv")
    result_df.to_csv(result_path, index=False, encoding="utf-8-sig")

    metric_df = pd.DataFrame({
        "loss": [loss_name],
        "best_valid_loss": [best_val_loss],
        "mae_recorded_sales": [mae],
        "rmse_recorded_sales": [rmse],
        "mape_recorded_sales": [mape],
    })

    metric_path = os.path.join(output_dir, f"metrics_recorded_sales_{loss_name}.csv")
    metric_df.to_csv(metric_path, index=False, encoding="utf-8-sig")

    # =========================
    # plot prediction
    # =========================
    plt.figure(figsize=(12, 6))
    plt.plot(result_df["date"], result_df["actual"], label="Actual")
    plt.plot(result_df["date"], result_df["prediction"], label="Prediction")

    plt.xlabel("date")
    plt.ylabel("recorded_sales")
    plt.title(f"Transformer Actual vs Prediction: {pred_len}-Day Forecast ({loss_name})")

    metric_text = (
        f"MAE  = {mae:.3f}\n"
        f"RMSE = {rmse:.3f}\n"
        f"MAPE = {mape:.2f}%"
    )

    plt.text(
        0.02,
        0.95,
        metric_text,
        transform=plt.gca().transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            alpha=0.8,
        ),
    )

    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    graph_path = os.path.join(output_dir, f"actual_vs_prediction_{loss_name}.png")
    plt.savefig(graph_path)
    plt.close()

    print(f"Saved result to: {result_path}")
    print(f"Saved graph to: {graph_path}")
    print(f"Saved loss history to: {loss_csv_path}")
    print(f"Saved loss graph to: {loss_graph_path}")
    print(f"Saved metrics to: {metric_path}")