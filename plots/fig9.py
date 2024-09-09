import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from constants import (
    COLOR_PALETTE,
    HUE_ORDER,
    LEGEND_FONTSIZE,
    NAME_FT,
    NAME_NMC_EX,
    NAME_NMC_FULL,
    PLOT_LINEWIDTH,
    TEXT_FONTSIZE,
    TICK_FONTSIZE,
)
from tqdm import tqdm
from wandb import Api

sns.set_style("darkgrid")


def parse_run(run, num_tasks):
    seed = run.config["seed"]
    run_name = run.group

    # download all the values of 'cont_eval_acc_tag/t_0' from the run
    metric_name = "test/avg_acc_tag"
    cont_eval = run.history(keys=[("%s" % metric_name)], samples=100000)[metric_name]
    max_steps = len(cont_eval)
    steps_per_task = max_steps // num_tasks
    return [
        {
            "run_name": run_name,
            "seed": seed,
            "task": step+1,
            "acc": acc,
        }
        for step, acc in enumerate(cont_eval)
    ]


def main():
    assert "WANDB_API_KEY" in os.environ, "WANDB_API_KEY is not set"
    wandb_api_key = os.environ["WANDB_API_KEY"]
    wandb_entity = "stability-gap"
    wandb_project = "cl-teacher-adaptation-src"

    root = Path(__file__).parent
    output_dir = root / "plots"
    output_dir.mkdir(exist_ok=True, parents=True)
    output_path_png = output_dir / "fig9.png"
    output_path_pdf = output_dir / "fig9.pdf"

    # Filters for the runs
    tag = "slca"
    dataset = "birds"
    num_tasks = 10
    nepochs = 100
    exemplars = [2, 5, 10]
    approaches = ["ft_nmc", "finetuning"]

    # Get all runs for the plots from wandb server"
    api = Api(api_key=wandb_api_key)
    runs = api.runs(
        path=f"{wandb_entity}/{wandb_project}",
        filters={
            "tags": tag,
            "config.datasets": [dataset],
            "config.num_tasks": num_tasks,
            "config.nepochs": nepochs,
            "config.approach": {"$in": approaches},
            "config.num_exemplars_per_class": {"$in": exemplars},
            "config.slca": True,
            "state": "finished",
        },
    )
    runs = list(runs)

    print(len(runs))

    # Parse runs to plotting format
    parsed_runs = [
        parse_run(r, num_tasks=num_tasks)
        for r in tqdm(runs, total=len(runs), desc="Loading wandb data...")
    ]
    flattened_runs = [point for r in parsed_runs for point in r]
    df = pd.DataFrame(flattened_runs)

    # Set names for the legend
    name_dict = {
        f"{dataset}_finetuning_t10s20_hz_m:2": "FT_2",
        f"{dataset}_ft_nmc_t10s20_hz_m:2_up:1": "NMC_2",
        f"{dataset}_finetuning_t10s20_hz_m:5": "FT_5",
        f"{dataset}_ft_nmc_t10s20_hz_m:5_up:1": "NMC_5",
        f"{dataset}_finetuning_t10s20_hz_m:10": "FT_10",
        f"{dataset}_ft_nmc_t10s20_hz_m:10_up:1": "NMC_10",
    }
    hue_dict = {
        "FT_2": 0,
        "NMC_2": 3,
        "FT_5": 1,
        "NMC_5": 4,
        "FT_10": 2,
        "NMC_10": 5,
    }
    color_dict = {
        "FT_2": "tab:red",
        "NMC_2": "tab:red",
        "FT_5": "tab:orange",
        "NMC_5": "tab:orange",
        "FT_10": "tab:blue",
        "NMC_10": "tab:blue",
    }
    dashes_dict = {
        "FT_2": (2, 0),
        "NMC_2": (3, 3),
        "FT_10": (2, 0),
        "NMC_10": (3, 3),
        "FT_5": (2, 0),
        "NMC_5": (3, 3),
    }

    df = df[df["run_name"].isin(name_dict.keys())]
    df["run_name"] = df["run_name"].map(name_dict)

    # Plot
    plt.figure()
    plt.clf()
    plt.cla()

    # Plot configuration
    xlabel = "Finished Task"
    ylabel = "Average Accuracy"
    title = "Birds + SL | 10 tasks"
    yticks = [20, 40, 60, 80, 100]

    plot = sns.lineplot(
        data=df,
        x="task",
        y="acc",
        hue="run_name",
        palette=color_dict,
        hue_order=hue_dict,
        style="run_name",
        dashes=dashes_dict,
        linewidth=1,
    )

    plot.set_title(title)
    plot.set_xlabel(xlabel)
    plt.xticks(range(1, num_tasks+1))
    plot.set_xlim(1, num_tasks)
    plot.set_ylabel(ylabel)
    # Set lower limit on y axis to 0
    plot.set_ylim(bottom=0)
    plot.set_yticks(yticks)

    # Set sizes for text and ticks
    plot.tick_params(axis="both", which="major", labelsize=TICK_FONTSIZE)
    plot.set_xlabel(xlabel, fontsize=TEXT_FONTSIZE)
    plot.set_ylabel(ylabel, fontsize=TEXT_FONTSIZE)
    plot.set_title(title, fontsize=TEXT_FONTSIZE)

    # Remove legend title and set fontsize
    # Reorder labels and handles for the legneds
    handles, labels = plot.get_legend_handles_labels()
    handles = [
        handles[labels.index("FT_2")],
        handles[labels.index("NMC_2")],
        handles[labels.index("FT_5")],
        handles[labels.index("NMC_5")],
        handles[labels.index("FT_10")],
        handles[labels.index("NMC_10")],
    ]
    labels = [
        "2 ex",
        '+ NMC',
        "5 ex",
        '+ NMC',
        "10 ex",
        '+ NMC',
    ]
    plot.legend(
        handles=handles,
        labels=labels,
        ncol=3,
        fontsize=12,
        title=None,
        loc="lower center",
        handlelength=1.5
    )

    # Save figure
    plt.tight_layout()
    plt.savefig(str(output_path_png), bbox_inches='tight', pad_inches=0)
    plt.savefig(str(output_path_pdf), bbox_inches='tight', pad_inches=0)


if __name__ == "__main__":
    main()
