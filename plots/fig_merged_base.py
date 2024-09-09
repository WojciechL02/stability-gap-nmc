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
    metric_name = "cont_eval_acc_tag/t_0"
    cont_eval = run.history(keys=[("%s" % metric_name)], samples=100000)[metric_name]
    max_steps = len(cont_eval)
    steps_per_task = max_steps // num_tasks
    return [
        {
            "run_name": run_name,
            "seed": seed,
            "task": step / steps_per_task,
            "acc": acc,
        }
        for step, acc in enumerate(cont_eval)
    ]


def plot_cifar100x5(ax, xlabel, ylabel, legend=False):
    assert "WANDB_API_KEY" in os.environ, "WANDB_API_KEY is not set"
    wandb_api_key = os.environ["WANDB_API_KEY"]
    wandb_entity = "stability-gap"
    wandb_project = "cl-teacher-adaptation-src"

    # Filters for the runs
    tag = "figure1"
    dataset = "cifar100_icarl"
    num_tasks = 5
    nepochs = 100
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
            "state": "finished",
        },
    )
    runs = list(runs)
    print(len(runs))
    # for r in runs:
    #     if "best_prototypes" in r.config.keys():
    #         if r.config["best_prototypes"] == True:
    #             r.group += "_full_set_prot"

    # Parse runs to plotting format
    parsed_runs = [
        parse_run(r, num_tasks=num_tasks)
        for r in tqdm(runs, total=len(runs), desc="Loading wandb data...")
    ]
    flattened_runs = [point for r in parsed_runs for point in r]
    df = pd.DataFrame(flattened_runs)

    # Set names for the legend
    name_dict = {
        "cifar100_icarl_finetuning_t5s20_hz_m:2000": NAME_FT,
        # "imagenet_subset_kaggle_ft_nmc_t5s20_hz_m:2000_up:1_full_set_prot": NAME_NMC_FULL,
        "cifar100_icarl_ft_nmc_t5s20_hz_m:2000_up:1": NAME_NMC_EX,
    }
    # print(df)
    df = df[df["run_name"].isin(name_dict.keys())]
    df["run_name"] = df["run_name"].map(name_dict)

    # Plot configuration
    title = f"CIFAR100 | {num_tasks} tasks"
    yticks = [10, 20, 30, 40, 50, 60]

    plot = sns.lineplot(
        data=df,
        x="task",
        y="acc",
        hue="run_name",
        palette=COLOR_PALETTE,
        hue_order=HUE_ORDER,
        linewidth=PLOT_LINEWIDTH,
        ax=ax,
        legend=False
    )
    plot.set_title(title)
    plot.set_xlabel(xlabel)
    plot.set_xticks(range(num_tasks + 1))
    plot.set_xlim(0, num_tasks)
    plot.set_ylabel(ylabel)
    # Set lower limit on y axis to 0
    plot.set_ylim(bottom=0)
    plot.set_yticks(yticks)

    # Set sizes for text and ticks
    plot.tick_params(axis="both", which="major", labelsize=TICK_FONTSIZE)
    plot.set_xlabel(xlabel, fontsize=TEXT_FONTSIZE)
    plot.set_ylabel(ylabel, fontsize=TEXT_FONTSIZE)
    plot.set_title(title, fontsize=TEXT_FONTSIZE)


def plot_cifar100x10(ax, xlabel, ylabel, legend=False):
    assert "WANDB_API_KEY" in os.environ, "WANDB_API_KEY is not set"
    wandb_api_key = os.environ["WANDB_API_KEY"]
    wandb_entity = "stability-gap"
    wandb_project = "cl-teacher-adaptation-src"

    # Filters for the runs
    tag = "figure1"
    dataset = "cifar100_icarl"
    num_tasks = 10
    nepochs = 100
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
            "state": "finished",
        },
    )
    runs = list(runs)
    print(len(runs))
    # for r in runs:
    #     if "best_prototypes" in r.config.keys():
    #         if r.config["best_prototypes"] == True:
    #             r.group += "_full_set_prot"

    # Parse runs to plotting format
    parsed_runs = [
        parse_run(r, num_tasks=num_tasks)
        for r in tqdm(runs, total=len(runs), desc="Loading wandb data...")
    ]
    flattened_runs = [point for r in parsed_runs for point in r]
    df = pd.DataFrame(flattened_runs)

    # Set names for the legend
    name_dict = {
        "cifar100_icarl_finetuning_t10s10_hz_m:2000": NAME_FT,
        # "imagenet_subset_kaggle_ft_nmc_t5s20_hz_m:2000_up:1_full_set_prot": NAME_NMC_FULL,
        "cifar100_icarl_ft_nmc_t10s10_hz_m:2000_up:1": NAME_NMC_EX,
    }
    # print(df)
    df = df[df["run_name"].isin(name_dict.keys())]
    df["run_name"] = df["run_name"].map(name_dict)

    # Plot configuration
    title = f"CIFAR100 | {num_tasks} tasks"
    yticks = [10, 20, 30, 40, 50, 60, 70]

    plot = sns.lineplot(
        data=df,
        x="task",
        y="acc",
        hue="run_name",
        palette=COLOR_PALETTE,
        hue_order=HUE_ORDER,
        linewidth=PLOT_LINEWIDTH,
        ax=ax,
        legend=False
    )
    plot.set_title(title)
    plot.set_xlabel(xlabel)
    plot.set_xticks(range(num_tasks + 1))
    plot.set_xlim(0, num_tasks)
    plot.set_ylabel(ylabel)
    # Set lower limit on y axis to 0
    plot.set_ylim(bottom=0)
    plot.set_yticks(yticks)

    # Set sizes for text and ticks
    plot.tick_params(axis="both", which="major", labelsize=TICK_FONTSIZE)
    plot.set_xlabel(xlabel, fontsize=TEXT_FONTSIZE)
    plot.set_ylabel(ylabel, fontsize=TEXT_FONTSIZE)
    plot.set_title(title, fontsize=TEXT_FONTSIZE)


def plot_in5(ax, xlabel, ylabel, legend=False):
    assert "WANDB_API_KEY" in os.environ, "WANDB_API_KEY is not set"
    wandb_api_key = os.environ["WANDB_API_KEY"]
    wandb_entity = "stability-gap"
    wandb_project = "cl-teacher-adaptation-src"

    # Filters for the runs
    tag = "figure1"
    dataset = "imagenet_subset_kaggle"
    num_tasks = 5
    nepochs = 100
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
            "state": "finished",
        },
    )
    runs = list(runs)
    print(len(runs))
    # for r in runs:
    #     if "best_prototypes" in r.config.keys():
    #         if r.config["best_prototypes"] == True:
    #             r.group += "_full_set_prot"

    # Parse runs to plotting format
    parsed_runs = [
        parse_run(r, num_tasks=num_tasks)
        for r in tqdm(runs, total=len(runs), desc="Loading wandb data...")
    ]
    flattened_runs = [point for r in parsed_runs for point in r]
    df = pd.DataFrame(flattened_runs)

    # Set names for the legend
    name_dict = {
        "imagenet_subset_kaggle_finetuning_t5s20_hz_m:2000": NAME_FT,
        # "imagenet_subset_kaggle_ft_nmc_t5s20_hz_m:2000_up:1_full_set_prot": NAME_NMC_FULL,
        "imagenet_subset_kaggle_ft_nmc_t5s20_hz_m:2000_up:1": NAME_NMC_EX,
    }
    # print(df)
    df = df[df["run_name"].isin(name_dict.keys())]
    df["run_name"] = df["run_name"].map(name_dict)

    # Plot configuration
    title = f"ImageNet100 | {num_tasks} tasks"
    yticks = [10, 20, 30, 40, 50, 60, 70, 80]

    plot = sns.lineplot(
        data=df,
        x="task",
        y="acc",
        hue="run_name",
        palette=COLOR_PALETTE,
        hue_order=HUE_ORDER,
        linewidth=PLOT_LINEWIDTH,
        ax=ax,
        legend=False
    )
    plot.set_title(title)
    plot.set_xlabel(xlabel)
    plot.set_xticks(range(num_tasks + 1))
    plot.set_xlim(0, num_tasks)
    plot.set_ylabel(ylabel)
    # Set lower limit on y axis to 0
    plot.set_ylim(bottom=0)
    plot.set_yticks(yticks)

    # Set sizes for text and ticks
    plot.tick_params(axis="both", which="major", labelsize=TICK_FONTSIZE)
    plot.set_xlabel(xlabel, fontsize=TEXT_FONTSIZE)
    plot.set_ylabel(ylabel, fontsize=TEXT_FONTSIZE)
    plot.set_title(title, fontsize=TEXT_FONTSIZE)


def plot_in10(ax, xlabel, ylabel, legend=False):
    assert "WANDB_API_KEY" in os.environ, "WANDB_API_KEY is not set"
    wandb_api_key = os.environ["WANDB_API_KEY"]
    wandb_entity = "stability-gap"
    wandb_project = "cl-teacher-adaptation-src"

    # Filters for the runs
    tag = "figure1"
    dataset = "imagenet_subset_kaggle"
    num_tasks = 10
    nepochs = 100
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
            "state": "finished",
        },
    )
    runs = list(runs)
    print(len(runs))
    # for r in runs:
    #     if "best_prototypes" in r.config.keys():
    #         if r.config["best_prototypes"] == True:
    #             r.group += "_full_set_prot"

    # Parse runs to plotting format
    parsed_runs = [
        parse_run(r, num_tasks=num_tasks)
        for r in tqdm(runs, total=len(runs), desc="Loading wandb data...")
    ]
    flattened_runs = [point for r in parsed_runs for point in r]
    df = pd.DataFrame(flattened_runs)

    # Set names for the legend
    name_dict = {
        "imagenet_subset_kaggle_finetuning_t10s10_hz_m:2000": NAME_FT,
        # "imagenet_subset_kaggle_ft_nmc_t5s20_hz_m:2000_up:1_full_set_prot": NAME_NMC_FULL,
        "imagenet_subset_kaggle_ft_nmc_t10s10_hz_m:2000_up:1": NAME_NMC_EX,
    }
    # print(df)
    df = df[df["run_name"].isin(name_dict.keys())]
    df["run_name"] = df["run_name"].map(name_dict)

    # Plot configuration
    title = f"ImageNet100 | {num_tasks} tasks"
    yticks = [10, 20, 30, 40, 50, 60, 70, 80, 90]

    plot = sns.lineplot(
        data=df,
        x="task",
        y="acc",
        hue="run_name",
        palette=COLOR_PALETTE,
        hue_order=HUE_ORDER,
        linewidth=PLOT_LINEWIDTH,
        ax=ax
    )
    plot.set_title(title)
    plot.set_xlabel(xlabel)
    plot.set_xticks(range(num_tasks + 1))
    plot.set_xlim(0, num_tasks)
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
        handles[labels.index(NAME_FT)],
        handles[labels.index(NAME_NMC_EX)],
    ]
    plot.legend(
        handles=handles,
        labels=labels,
        loc="upper right",
        fontsize=LEGEND_FONTSIZE,
        title=None,
    )


def main():
    root = Path(__file__).parent
    output_dir = root / "plots"
    output_dir.mkdir(exist_ok=True, parents=True)

    fig, axes = plt.subplots(1, 4, figsize=(25.60, 4.80))  # , width_ratios=[1, 1, 1, 1]

    os.environ['WANDB_API_KEY'] = '434fcc1957118a52a224c4d4a88db52186983f58'

    plot_cifar100x5(axes[0], xlabel="Finished Task", ylabel="Task 1 Accuracy")
    plot_cifar100x10(axes[1], xlabel="Finished Task", ylabel=None)
    plot_in5(axes[2], xlabel='Finished Task', ylabel=None)
    plot_in10(axes[3], xlabel='Finished Task', ylabel=None)

    output_path_png = output_dir / "fig_merged_base.png"
    output_path_pdf = output_dir / "fig_merged_base.pdf"
    plt.tight_layout()
    plt.savefig(str(output_path_png))
    plt.savefig(str(output_path_pdf))


if __name__ == "__main__":
    main()
