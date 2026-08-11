import glob
import warnings
from pathlib import Path

import pandas as pd
from rich import print


def get_config_paths(
    dataset, metric, checkpoint_dir="l_checkpoints", config_dir="config_files"
):
    checkpoint_path_list = glob.glob(
        f"{checkpoint_dir}/{dataset}/*/*/best_{metric}_val_metrics/*"
    )

    config_path_list = []
    for path in checkpoint_path_list:
        model, version = path.split("/")[2:4]
        config_path = f"{config_dir}/{version}/{model}_{dataset.lower()}_{version}.yaml"
        config_path_list.append(config_path)

    return config_path_list, checkpoint_path_list


def find_untested_models(dataset, metric, analysis_dir="Analysis"):
    current_analysis_path = Path(f"{analysis_dir}/{dataset}.csv")
    if current_analysis_path.exists():
        print(f"[bold green] Loading {dataset} Analysis File. [/bold green]")
        current_df = pd.read_csv(current_analysis_path)
    else:
        print(f"[bold green] File {current_analysis_path} not found. [/bold green]")
        warnings.warn("Legacy analysis file not found. Creating a new one.")
        current_df = pd.DataFrame(
            columns=[
                "model",
                "dataset",
                "version",
                "losses/val_loss",
                "val_metrics/accuracy",
                "val_metrics/f1",
                "val_metrics/miou",
                "val_metrics/precision",
                "val_metrics/recall",
                "losses/test_loss",
                "test_metrics/accuracy",
                "test_metrics/f1",
                "test_metrics/miou",
                "test_metrics/precision",
                "test_metrics/recall",
            ]
        )

    config_path, checkpoint_path = get_config_paths(dataset, metric)

    dict_list = dict(
        dataset=[], model=[], version=[], config_path=[], checkpoint_path=[]
    )
    for config, checkpoint in zip(config_path, checkpoint_path):
        name, model, version = checkpoint.lower().split("/")[1:4]
        dict_list["dataset"].append(name)
        dict_list["model"].append(model)
        dict_list["version"].append(version)
        dict_list["config_path"].append(config)
        dict_list["checkpoint_path"].append(checkpoint)

    new_df = pd.DataFrame(dict_list)
    diff = (
        current_df[["model", "dataset", "version"]]
        .merge(new_df, how="outer", indicator=True)
        .query("_merge=='right_only'")
    )
    config_path, checkpoint_path = (
        diff["config_path"].tolist(),
        diff["checkpoint_path"].tolist(),
    )
    print(f"[bold yellow] Returning {len(config_path)} untested models [/bold yellow]")
    return config_path, checkpoint_path, current_df
