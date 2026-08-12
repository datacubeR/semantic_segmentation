import argparse

import pandas as pd
from rich import print

from .analysis import (
    create_results_dataframe,
    find_untested_models,
    test_gridsegmentor_checkpoint,
    test_simplesegmentor_checkpoint,
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    type=str,
    default="Vaihingen",
    choices=["DeadTrees", "LoveDA", "Vaihingen", "Potsdam"],
)
parser.add_argument(
    "--metric",
    type=str,
    default="miou",
    choices=["miou", "f1", "precision", "recall", "accuracy"],
)
args = parser.parse_args()

dataset = args.dataset
metric = args.metric

config_path, checkpoint_path, current_df = find_untested_models(dataset, metric)
if dataset.lower() in ["vaihingen", "potsdam"]:
    testing_function = test_gridsegmentor_checkpoint

elif dataset.lower() in ["loveda", "deadtrees"]:
    testing_function = test_simplesegmentor_checkpoint

else:
    raise ValueError(
        f"Unknown dataset: {dataset}. There are no testing functions for this Dataset."
    )

new_results_df = create_results_dataframe(
    checkpoint_path, config_path, testing_function
)


print("[bold blue] Saving Results to CSV [/bold blue]")
(
    pd.concat([current_df, new_results_df])
    .reset_index(drop=True)
    .sort_values(by=["version", f"test_metrics/{metric}"], ascending=[True, False])
    .to_csv(f"Analysis/{dataset}.csv", index=False)
)
