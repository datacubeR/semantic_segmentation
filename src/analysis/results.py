import warnings
from pathlib import Path

import pandas as pd
from rich import print


def create_results_dataframe(checkpoint_path: str, config_path: str, testing_function):
    warnings.filterwarnings("ignore")

    warnings.filterwarnings(
        "ignore",
        message="You called `self.log",
    )
    validation_list = []
    for checkpoint, config in zip(checkpoint_path, config_path):
        model, dataset, version = Path(config).stem.split("_")
        print(f"[bold red]Using config: {config} [/bold red]")
        print(f"[bold green]Validating checkpoint: {checkpoint} [/bold green]")
        trainer, segmentation_model, dm = testing_function(checkpoint, config)
        validation = trainer.validate(
            model=segmentation_model, datamodule=dm, verbose=False
        )
        test = trainer.test(model=segmentation_model, datamodule=dm, verbose=False)
        final_output = (
            dict(model=model, dataset=dataset, version=version)
            | validation[0]
            | test[0]
        )
        validation_list.append(final_output)

    df = pd.DataFrame(validation_list)
    print(f"[bold yellow] Returning {len(df)} new results [/bold yellow]")
    return df
