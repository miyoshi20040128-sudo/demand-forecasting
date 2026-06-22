import pandas as pd
import numpy as np
import torch

from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


class DemandDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_and_merge_data(sales_path, weather_path, area):
    sales = pd.read_csv(sales_path)
    weather = pd.read_csv(weather_path)

    sales["date"] = pd.to_datetime(sales["date"])
    weather["date"] = pd.to_datetime(weather["date"])

    sales["item_index"] = sales["item_index"].astype(str)
    if "item_index" in weather.columns:
        weather["item_index"] = weather["item_index"].astype(str)

    if area is not None:
        sales = sales[sales["area"] == area].copy()
        weather = weather[weather["area"] == area].copy()

    # sales側は true_sales を使わない
    if "true_sales" in sales.columns:
        sales = sales.drop(columns=["true_sales"])

    # weather側に true_sales / recorded_sales がある場合は、特徴量として使わない
    # recorded_sales は sales側の目的変数と被るので消す
    drop_weather_cols = []
    for col in ["true_sales", "recorded_sales"]:
        if col in weather.columns:
            drop_weather_cols.append(col)

    if len(drop_weather_cols) > 0:
        weather = weather.drop(columns=drop_weather_cols)

    merge_keys = ["area", "date"]

    if "item_index" in sales.columns and "item_index" in weather.columns:
        merge_keys.append("item_index")

    merged = pd.merge(
        sales,
        weather,
        on=merge_keys,
        how="left",
    )

    merged["item_index"] = merged["item_index"].astype(str)
    merged["year"] = merged["date"].dt.year

    merged = merged.sort_values(["item_index", "date"]).reset_index(drop=True)

    return merged


def add_time_features(df):
    df = df.copy()

    df["date"] = pd.to_datetime(df["date"])

    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    return df


def encode_item_index(train_df, valid_df):
    train_df = train_df.copy()
    valid_df = valid_df.copy()

    item_values = sorted(train_df["item_index"].astype(str).unique())
    item_map = {v: i for i, v in enumerate(item_values)}

    train_df["item_code"] = train_df["item_index"].astype(str).map(item_map)
    valid_df["item_code"] = valid_df["item_index"].astype(str).map(item_map).fillna(-1)

    return train_df, valid_df, item_map


def encode_categorical_weather_columns(train_df, valid_df, exclude_cols):
    train_df = train_df.copy()
    valid_df = valid_df.copy()

    categorical_cols = []

    for col in train_df.columns:
        if col in exclude_cols:
            continue

        if train_df[col].dtype == "object":
            categorical_cols.append(col)

    category_maps = {}

    for col in categorical_cols:
        train_df[col] = train_df[col].astype(str)
        valid_df[col] = valid_df[col].astype(str)

        values = sorted(train_df[col].unique())
        mapping = {v: i for i, v in enumerate(values)}

        code_col = f"{col}_code"

        train_df[code_col] = train_df[col].map(mapping)
        valid_df[code_col] = valid_df[col].map(mapping).fillna(-1)

        category_maps[col] = mapping

    return train_df, valid_df, categorical_cols, category_maps


def select_last_n_months(df, n_months):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    months = (
        df["date"]
        .dt.to_period("M")
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    if n_months > len(months):
        raise ValueError(
            f"validation_months={n_months} ですが、"
            f"検証年のデータには {len(months)} ヶ月しかありません。"
        )

    target_months = months[-n_months:]

    target_df = df[
        df["date"].dt.to_period("M").isin(target_months)
    ].copy()

    target_df = target_df.sort_values("date").reset_index(drop=True)

    return target_df


def create_windows_from_values(values, target_idx, seq_len, pred_len, stride):
    X = []
    y = []

    max_start = len(values) - seq_len - pred_len + 1

    for start in range(0, max_start, stride):
        encoder_start = start
        encoder_end = start + seq_len

        decoder_start = encoder_end
        decoder_end = decoder_start + pred_len

        X.append(values[encoder_start:encoder_end])
        y.append(values[decoder_start:decoder_end, target_idx])

    return np.array(X), np.array(y)


def create_train_windows_by_item_year(
    train_df,
    scaler,
    feature_cols,
    target_idx,
    seq_len,
    pred_len,
):
    X_list = []
    y_list = []

    grouped = train_df.groupby(["item_index", "year"])

    for (item_index, year), group_df in grouped:
        group_df = group_df.sort_values("date").reset_index(drop=True)

        if len(group_df) < seq_len + pred_len:
            continue

        values = scaler.transform(group_df[feature_cols])

        # ここが重要
        # pred_len=7なら、7日ずつずらして、予測対象が被らないようにする
        X_group, y_group = create_windows_from_values(
            values=values,
            target_idx=target_idx,
            seq_len=seq_len,
            pred_len=pred_len,
            stride=pred_len,
        )

        X_list.append(X_group)
        y_list.append(y_group)

    if len(X_list) == 0:
        raise ValueError(
            "学習サンプルを作れませんでした。"
            "seq_len や pred_len が長すぎる可能性があります。"
        )

    X_train = np.concatenate(X_list, axis=0)
    y_train = np.concatenate(y_list, axis=0)

    return X_train, y_train


def create_validation_windows(
    train_df,
    valid_df,
    valid_item_index,
    scaler,
    feature_cols,
    target_idx,
    seq_len,
    pred_len,
    validation_months,
):
    valid_item_index = str(valid_item_index)

    train_item_df = train_df[
        train_df["item_index"].astype(str) == valid_item_index
    ].copy()

    valid_item_df = valid_df[
        valid_df["item_index"].astype(str) == valid_item_index
    ].copy()

    if len(valid_item_df) == 0:
        raise ValueError(f"valid_year に item_index={valid_item_index} のデータがありません。")

    valid_item_df = valid_item_df.sort_values("date").reset_index(drop=True)

    valid_target_df = select_last_n_months(
        valid_item_df,
        validation_months,
    )

    target_start_date = valid_target_df["date"].min()

    # 検証対象より前のvalid年内データ
    valid_before_target_df = valid_item_df[
        valid_item_df["date"] < target_start_date
    ].copy()

    valid_before_target_df = valid_before_target_df.sort_values("date").reset_index(drop=True)

    # valid年内の直前履歴
    context_from_valid = valid_before_target_df.tail(seq_len).copy()

    # 足りない分は学習年の最後から補う
    need_from_train = max(0, seq_len - len(context_from_valid))

    if need_from_train > 0:
        train_context = (
            train_item_df
            .sort_values("date")
            .tail(need_from_train)
            .copy()
        )
    else:
        train_context = train_item_df.iloc[0:0].copy()

    train_context["is_validation_target"] = False
    context_from_valid["is_validation_target"] = False
    valid_target_df["is_validation_target"] = True

    validation_full_df = pd.concat(
        [
            train_context,
            context_from_valid,
            valid_target_df,
        ],
        axis=0,
        ignore_index=True,
    )

    validation_full_df = validation_full_df.reset_index(drop=True)

    values = scaler.transform(validation_full_df[feature_cols])

    X_val = []
    y_val = []
    val_dates = []
    val_actuals = []
    val_window_ids = []
    val_horizons = []

    max_start = len(values) - seq_len - pred_len + 1

    window_id = 0

    # ここも重要
    # pred_len=7なら、7日ずつ進める
    for start in range(0, max_start, pred_len):
        encoder_start = start
        encoder_end = start + seq_len

        decoder_start = encoder_end
        decoder_end = decoder_start + pred_len

        target_flags = validation_full_df.loc[
            decoder_start:decoder_end - 1,
            "is_validation_target"
        ]

        # 7日分すべてが検証対象に入っているときだけ採用
        # これにより、途中で被ったり、最後の半端な数日だけ予測したりしない
        if target_flags.all():
            X_window = values[encoder_start:encoder_end]
            y_window = values[decoder_start:decoder_end, target_idx]

            X_val.append(X_window)
            y_val.append(y_window)

            dates = validation_full_df.loc[
                decoder_start:decoder_end - 1,
                "date"
            ].tolist()

            actuals = validation_full_df.loc[
                decoder_start:decoder_end - 1,
                "recorded_sales"
            ].tolist()

            val_dates.extend(dates)
            val_actuals.extend(actuals)

            for h in range(1, pred_len + 1):
                val_window_ids.append(window_id)
                val_horizons.append(h)

            window_id += 1

    if len(X_val) == 0:
        raise ValueError(
            "検証サンプルを作れませんでした。"
            "seq_len, pred_len, validation_months を確認してください。"
        )

    X_val = np.array(X_val)
    y_val = np.array(y_val)

    return (
        X_val,
        y_val,
        pd.to_datetime(val_dates),
        np.array(val_actuals, dtype=float),
        np.array(val_window_ids),
        np.array(val_horizons),
        validation_full_df,
    )


def create_datasets(config):
    sales_path = config["path"]["sales_path"]
    weather_path = config["path"]["weather_path"]

    area = config["data"]["area"]
    train_years = config["data"]["train_years"]
    valid_year = config["data"]["valid_year"]
    valid_item_index = config["data"]["valid_item_index"]
    validation_months = config["data"]["validation_months"]

    seq_len = config["model"]["seq_len"]
    pred_len = config["model"]["pred_len"]

    df = load_and_merge_data(
        sales_path=sales_path,
        weather_path=weather_path,
        area=area,
    )

    df = add_time_features(df)

    train_df = df[df["year"].isin(train_years)].copy()
    valid_df = df[df["year"] == valid_year].copy()

    if len(train_df) == 0:
        raise ValueError("train_df が空です。train_years と date の年を確認してください。")

    if len(valid_df) == 0:
        raise ValueError("valid_df が空です。valid_year と date の年を確認してください。")

    train_df, valid_df, item_map = encode_item_index(train_df, valid_df)

    exclude_cols = [
        "area",
        "date",
        "item_index",
        "year",
        "recorded_sales",
    ]

    train_df, valid_df, categorical_cols, category_maps = encode_categorical_weather_columns(
        train_df=train_df,
        valid_df=valid_df,
        exclude_cols=exclude_cols,
    )

    base_feature_cols = [
        "recorded_sales",
        "day_of_week",
        "month",
        "day",
        "is_weekend",
        "item_code",
    ]

    # weather.csv 側に数値特徴量がある場合は自動で入れる
    numeric_extra_cols = []

    for col in train_df.columns:
        if col in exclude_cols:
            continue

        if col in base_feature_cols:
            continue

        if col.endswith("_code"):
            continue

        if pd.api.types.is_numeric_dtype(train_df[col]):
            numeric_extra_cols.append(col)

    categorical_code_cols = [f"{col}_code" for col in categorical_cols]

    feature_cols = base_feature_cols + numeric_extra_cols + categorical_code_cols

    target_col = "recorded_sales"
    target_idx = feature_cols.index(target_col)

    for col in feature_cols:
        train_df[col] = train_df[col].astype(float)
        valid_df[col] = valid_df[col].astype(float)

    train_df = train_df.dropna(subset=feature_cols).reset_index(drop=True)
    valid_df = valid_df.dropna(subset=feature_cols).reset_index(drop=True)

    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols])

    X_train, y_train = create_train_windows_by_item_year(
        train_df=train_df,
        scaler=scaler,
        feature_cols=feature_cols,
        target_idx=target_idx,
        seq_len=seq_len,
        pred_len=pred_len,
    )

    (
        X_val,
        y_val,
        val_dates,
        val_actuals,
        val_window_ids,
        val_horizons,
        validation_full_df,
    ) = create_validation_windows(
        train_df=train_df,
        valid_df=valid_df,
        valid_item_index=valid_item_index,
        scaler=scaler,
        feature_cols=feature_cols,
        target_idx=target_idx,
        seq_len=seq_len,
        pred_len=pred_len,
        validation_months=validation_months,
    )

    train_dataset = DemandDataset(X_train, y_train)
    valid_dataset = DemandDataset(X_val, y_val)

    info = {
        "df": df,
        "train_df": train_df,
        "valid_df": valid_df,
        "validation_full_df": validation_full_df,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "target_col": target_col,
        "target_idx": target_idx,
        "val_dates": val_dates,
        "val_actuals": val_actuals,
        "val_window_ids": val_window_ids,
        "val_horizons": val_horizons,
        "item_map": item_map,
        "category_maps": category_maps,
    }

    print("===== dataset info =====")
    print(f"area: {area}")
    print(f"train_years: {train_years}")
    print(f"valid_year: {valid_year}")
    print(f"valid_item_index: {valid_item_index}")
    print(f"validation_months: {validation_months}")
    print(f"seq_len: {seq_len}")
    print(f"pred_len: {pred_len}")
    print(f"feature_cols: {feature_cols}")
    print(f"train rows: {len(train_df)}")
    print(f"valid rows: {len(valid_df)}")
    print(f"train samples: {len(train_dataset)}")
    print(f"valid samples: {len(valid_dataset)}")
    print(f"valid predicted days: {len(val_dates)}")
    print(f"valid date range: {val_dates.min()} ~ {val_dates.max()}")
    print("========================")

    return train_dataset, valid_dataset, info