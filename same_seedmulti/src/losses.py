import torch
import torch.nn as nn


def _pairwise_distances(y_pred, y_true):
    """
    y_pred: [batch, pred_len]
    y_true: [batch, pred_len]

    return:
        D: [batch, pred_len, pred_len]
        D[b, i, j] = (y_pred[b, i] - y_true[b, j])^2
    """
    y_pred = y_pred.unsqueeze(-1)
    y_true = y_true.unsqueeze(-1)

    D = (y_pred - y_true.transpose(1, 2)) ** 2

    return D


def _soft_dtw(D, gamma):
    """
    Soft-DTW の forward 計算。

    D: [batch, N, M]
    gamma: smoothing parameter

    return:
        soft_dtw_value: [batch]
    """
    batch_size, N, M = D.shape
    device = D.device
    dtype = D.dtype

    R = torch.full(
        (batch_size, N + 1, M + 1),
        float("inf"),
        device=device,
        dtype=dtype,
    )

    R[:, 0, 0] = 0.0

    for i in range(1, N + 1):
        for j in range(1, M + 1):
            r0 = R[:, i - 1, j]
            r1 = R[:, i, j - 1]
            r2 = R[:, i - 1, j - 1]

            r = torch.stack([r0, r1, r2], dim=-1)

            soft_min = -gamma * torch.logsumexp(-r / gamma, dim=-1)

            R[:, i, j] = D[:, i - 1, j - 1] + soft_min

    return R[:, N, M]


class DILATELoss(nn.Module):
    """
    DILATE loss の簡易PyTorch実装。

    loss = alpha * shape_loss + (1 - alpha) * temporal_loss

    shape_loss:
        Soft-DTW

    temporal_loss:
        Soft-DTWのアラインメント行列を使って、
        予測系列と正解系列の時間的ズレを罰する。

    注意:
        論文の高速custom backward実装ではなく、
        PyTorch autogradを使った簡易実装。
        pred_len=7くらいならまず試せる。
    """

    def __init__(self, alpha=0.5, gamma=0.1):
        super().__init__()

        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha must be between 0 and 1.")

        if gamma <= 0:
            raise ValueError("gamma must be positive.")

        self.alpha = alpha
        self.gamma = gamma

    def forward(self, y_pred, y_true):
        """
        y_pred: [batch, pred_len]
        y_true: [batch, pred_len]
        """
        if y_pred.dim() != 2:
            raise ValueError(
                f"y_pred must be 2D [batch, pred_len], got shape {y_pred.shape}"
            )

        if y_true.dim() != 2:
            raise ValueError(
                f"y_true must be 2D [batch, pred_len], got shape {y_true.shape}"
            )

        if y_pred.shape != y_true.shape:
            raise ValueError(
                f"y_pred and y_true must have same shape, "
                f"got {y_pred.shape} and {y_true.shape}"
            )

        batch_size, pred_len = y_pred.shape
        device = y_pred.device
        dtype = y_pred.dtype

        # D[b, i, j] = (pred_i - true_j)^2
        D = _pairwise_distances(y_pred, y_true)

        # shape loss = Soft-DTW
        shape_loss_each = _soft_dtw(D, gamma=self.gamma)
        shape_loss = shape_loss_each.mean()

        # Soft-DTWをDで微分すると soft alignment matrix になる
        alignment = torch.autograd.grad(
            outputs=shape_loss_each.sum(),
            inputs=D,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        # 時間ズレペナルティ行列 Omega
        # Omega[i, j] = ((i - j)^2) / pred_len^2
        time_index = torch.arange(pred_len, device=device, dtype=dtype)

        omega = (time_index[:, None] - time_index[None, :]) ** 2
        omega = omega / (pred_len ** 2)

        # temporal loss = <A_gamma, Omega>
        temporal_loss_each = torch.sum(alignment * omega, dim=(1, 2))
        temporal_loss = temporal_loss_each.mean()

        loss = self.alpha * shape_loss + (1.0 - self.alpha) * temporal_loss

        return loss


class DerivativeRegularizedMSELoss(nn.Module):
    """
    CONTIME論文の損失関数を、Transformer用に差分で近似したもの。

    loss = alpha * MSE(y_pred, y_true)
         + beta  * MSE(diff(y_pred), diff(y_true))

    y_pred: [batch, pred_len]
    y_true: [batch, pred_len]

    例:
        y_pred = [100, 110, 105, 120]
        diff   = [10, -5, 15]

    値そのものだけでなく、上がり方・下がり方も一致させる。
    """

    def __init__(self, alpha=1.0, beta=0.1):
        super().__init__()

        if alpha < 0:
            raise ValueError("alpha must be non-negative.")

        if beta < 0:
            raise ValueError("beta must be non-negative.")

        self.alpha = alpha
        self.beta = beta
        self.mse = nn.MSELoss()

    def forward(self, y_pred, y_true):
        """
        y_pred: [batch, pred_len]
        y_true: [batch, pred_len]
        """
        if y_pred.dim() != 2:
            raise ValueError(
                f"y_pred must be 2D [batch, pred_len], got shape {y_pred.shape}"
            )

        if y_true.dim() != 2:
            raise ValueError(
                f"y_true must be 2D [batch, pred_len], got shape {y_true.shape}"
            )

        if y_pred.shape != y_true.shape:
            raise ValueError(
                f"y_pred and y_true must have same shape, "
                f"got {y_pred.shape} and {y_true.shape}"
            )

        # 値そのもののMSE
        task_loss = self.mse(y_pred, y_true)

        # pred_len=1 の場合は差分が作れないので、通常のMSEだけ返す
        if y_pred.shape[1] < 2:
            return task_loss

        # 時間方向の差分
        pred_diff = y_pred[:, 1:] - y_pred[:, :-1]
        true_diff = y_true[:, 1:] - y_true[:, :-1]

        # 増減のMSE
        derivative_loss = self.mse(pred_diff, true_diff)

        loss = self.alpha * task_loss + self.beta * derivative_loss

        return loss


def get_loss_function(loss_name, config=None):
    """
    config.toml から損失関数を切り替えるための関数。

    使える loss_name:
        "mse"
        "mae"
        "huber"
        "dilate"
        "derivative"
    """
    loss_name = loss_name.lower()

    if loss_name == "mse":
        return nn.MSELoss()

    if loss_name == "mae":
        return nn.L1Loss()

    if loss_name == "huber":
        delta = 1.0

        if config is not None and "loss" in config:
            delta = config["loss"].get("huber_delta", delta)

        return nn.HuberLoss(delta=delta)

    if loss_name == "dilate":
        alpha = 0.5
        gamma = 0.1

        if config is not None and "loss" in config:
            alpha = config["loss"].get("alpha", alpha)
            gamma = config["loss"].get("gamma", gamma)

        return DILATELoss(alpha=alpha, gamma=gamma)

    if loss_name == "derivative":
        alpha = 1.0
        beta = 0.1

        if config is not None and "loss" in config:
            alpha = config["loss"].get("derivative_alpha", alpha)
            beta = config["loss"].get("derivative_beta", beta)

        return DerivativeRegularizedMSELoss(alpha=alpha, beta=beta)

    raise ValueError(
        f"Unknown loss function: {loss_name}. "
        "Choose from 'mse', 'mae', 'huber', 'dilate', 'derivative'."
    )