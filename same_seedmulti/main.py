import copy
import tomllib

from src.train import train
from compare_all_losses import create_combined_report


def load_config(config_path="config.toml"):
    with open(config_path, "rb") as f:
        return tomllib.load(f)


def main():
    config = load_config()

    loss_names = config.get("compare", {}).get(
        "loss_names",
        ["mse", "mae", "huber", "dilate", "derivative"],
    )

    print("===== Run all loss functions =====")
    print(f"loss_names: {loss_names}")
    print("==================================")

    for loss_name in loss_names:
        print()
        print("=" * 60)
        print(f"Start training with loss = {loss_name}")
        print("=" * 60)

        run_config = copy.deepcopy(config)
        run_config["train"]["loss"] = loss_name

        train(run_config)

    print()
    print("=" * 60)
    print("Create combined comparison report")
    print("=" * 60)

    create_combined_report(config)

    print()
    print("All done.")


if __name__ == "__main__":
    main()