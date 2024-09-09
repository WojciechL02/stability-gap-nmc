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
    output_path_png = output_dir / "fig7.png"
    output_path_pdf = output_dir / "fig7.pdf"

    # Filters for the runs
    tag = "figure1"
    dataset = "cifar100_icarl"
    num_tasks = 10
    nepochs = 100
    exemplars = [100, 200, 500, 1000, 2000, 5000]
    approaches = ["ft_nmc", "finetuning"]
    no_names = ["chocolate-paper-1525", "soft-butterfly-1525", "good-microwave-1525"]

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
            "config.num_exemplars": {"$in": exemplars},
            "state": "finished",
            "created_at": {"$gt": "2024-08-20T01"},
        },
    )
    runs = list(runs)
    runs = [r for r in runs if r.name not in no_names]

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
        "cifar100_icarl_finetuning_t10s10_hz_m:100": "FT_100",
        "cifar100_icarl_ft_nmc_t10s10_hz_m:100_up:1": "NMC_100",
        "cifar100_icarl_finetuning_t10s10_hz_m:500": "FT_500",
        "cifar100_icarl_ft_nmc_t10s10_hz_m:500_up:1": "NMC_500",
        "cifar100_icarl_finetuning_t10s10_hz_m:1000": "FT_1000",
        "cifar100_icarl_ft_nmc_t10s10_hz_m:1000_up:1": "NMC_1000",
        "cifar100_icarl_finetuning_t10s10_hz_m:2000": "FT_2000",
        "cifar100_icarl_ft_nmc_t10s10_hz_m:2000_up:1": "NMC_2000",
        "cifar100_icarl_finetuning_t10s10_hz_m:5000": "FT_5000",
        "cifar100_icarl_ft_nmc_t10s10_hz_m:5000_up:1": "NMC_5000",
    }
    hue_dict = {
        "FT_100": 0,
        "NMC_100": 5,
        "FT_500": 1,
        "NMC_500": 6,
        "FT_1000": 2,
        "NMC_1000": 7,
        "FT_2000": 3,
        "NMC_2000": 8,
        "FT_5000": 4,
        "NMC_5000": 9,
    }
    color_dict = {
        "FT_100": "tab:gray",
        "NMC_100": "tab:gray",
        "FT_500": "tab:red",
        "NMC_500": "tab:red",
        "FT_1000": "tab:orange",
        "NMC_1000": "tab:orange",
        "FT_2000": "tab:blue",
        "NMC_2000": "tab:blue",
        "FT_5000": "tab:green",
        "NMC_5000": "tab:green",
    }
    dashes_dict = {
        "FT_100": (2, 0),
        "NMC_100": (3, 3),
        "FT_500": (2, 0),
        "NMC_500": (3, 3),
        "FT_1000": (2, 0),
        "NMC_1000": (3, 3),
        "FT_2000": (2, 0),
        "NMC_2000": (3, 3),
        "FT_5000": (2, 0),
        "NMC_5000": (3, 3),
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
    title = "CIFAR100 | 10 tasks"
    yticks = [10, 20, 30, 40, 50, 60, 70]

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
        handles[labels.index("FT_100")],
        handles[labels.index("FT_500")],
        handles[labels.index("FT_1000")],
        handles[labels.index("FT_2000")],
        handles[labels.index("FT_5000")],
        handles[labels.index("NMC_100")],
        handles[labels.index("NMC_500")],
        handles[labels.index("NMC_1000")],
        handles[labels.index("NMC_2000")],
        handles[labels.index("NMC_5000")],
    ]
    labels = [
        "100 ex",
        "500 ex",
        "1000 ex",
        "2000 ex",
        "5000 ex",
        '+ NMC',
        '+ NMC',
        '+ NMC',
        '+ NMC',
        '+ NMC',
    ]
    plot.legend(
        handles=handles,
        labels=labels,
        ncol=2,
        fontsize=12,
        title=None,
        loc="upper right",
        handlelength=1.5,
        columnspacing=1.0
    )

    # Save figure
    plt.tight_layout()
    plt.savefig(str(output_path_png))
    plt.savefig(str(output_path_pdf))


if __name__ == "__main__":
    main()
