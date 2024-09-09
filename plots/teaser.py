import os
from pathlib import Path

import matplotlib.lines
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


def main():
    # os.environ['WANDB_API_KEY'] = '434fcc1957118a52a224c4d4a88db52186983f58'
    wandb_api_key = os.environ["WANDB_API_KEY"]
    wandb_entity = "stability-gap"
    wandb_project = "cl-teacher-adaptation-src"

    root = Path(__file__).parent
    output_dir = root / "plots"
    output_dir.mkdir(exist_ok=True, parents=True)
    output_path_png = output_dir / "teaser.png"
    output_path_pdf = output_dir / "teaser.pdf"

    # Filters for the runs
    tag = "figure1"
    dataset = "cifar100_icarl"
    num_tasks = 5
    nepochs = 100
    exemplars = 2000
    approaches = ["finetuning", 'ft_nmc']

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
            "config.num_exemplars": exemplars,
            "created_at": {"$gt": "2024-08-20T01"},
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
        "cifar100_icarl_finetuning_t5s20_hz_m:2000": NAME_FT,
        # "cifar100_icarl_ft_nmc_t5s20_hz_m:2000_up:1_full_set_prot": NAME_NMC_FULL,
        "cifar100_icarl_ft_nmc_t5s20_hz_m:2000_up:1": NAME_NMC_EX,
    }
    df = df[df["run_name"].isin(name_dict.keys())]
    df["run_name"] = df["run_name"].map(name_dict)
    df = df[df['task'] < 2]

    # Plot
    plt.figure()
    plt.clf()
    plt.cla()

    # Plot configuration
    xlabel = "Finished Task"
    ylabel = "Task 1 Accuracy"
    title = "CIFAR100 | 5 tasks"
    yticks = [10, 20, 30, 40, 50, 60, 70]

    plot = sns.lineplot(
        data=df,
        x="task",
        y="acc",
        hue="run_name",
        palette=COLOR_PALETTE,
        hue_order=HUE_ORDER,
        linewidth=PLOT_LINEWIDTH,
        legend=True
    )
    plot.set_title(title)
    plot.set_xlabel(xlabel)
    plt.xticks(range(num_tasks + 1))
    plot.set_xlim(0, 2)
    plot.set_ylabel(ylabel)
    # Set lower limit on y axis to 0
    plot.set_ylim(bottom=0)
    plot.set_yticks(yticks)

    # Set sizes for text and ticks
    plot.tick_params(axis="both", which="major", labelsize=TICK_FONTSIZE)
    plot.set_xlabel(xlabel, fontsize=TEXT_FONTSIZE)
    plot.set_ylabel(ylabel, fontsize=TEXT_FONTSIZE)
    plot.set_title(title, fontsize=TEXT_FONTSIZE)

    # Annotate for finetuning
    max = 62
    min = 22
    xpos = 1.2
    # Make a bidirectional arrow
    HEAD_WIDTH = 0.05
    HEAD_LENGTH = 1
    LINEWIDTH=1
    FONTSIZE = 16
    plt.arrow(xpos, min, 0, max - min, color='black', head_width=HEAD_WIDTH, head_length=HEAD_LENGTH,
              length_includes_head=True)
    plt.arrow(xpos, max, 0, min - max, color='black', head_width=HEAD_WIDTH, head_length=HEAD_LENGTH,
              length_includes_head=True)
    # Add annotation in the right angle close to the arrow
    # plot.annotate("stability gap", xy=(xpos+0.03, min + 13), xytext=(xpos+0.03, min + 13), rotation=90, fontsize=FONTSIZE)

    # Draw dashed lines from 1 to arrow heads
    # plt.axline((1, min), (xpos, min), marker=None, color='black', linestyle='dashed')
    # plt.axline((1, max), (xpos, max), marker=None, color='black', linestyle='dashed')
    plot.add_line(matplotlib.lines.Line2D([xpos, 1.05], [min, min], color='black', linestyle='dashed', linewidth=LINEWIDTH))
    plot.add_line(matplotlib.lines.Line2D([xpos, 1.0], [max, max], color='black', linestyle='dashed', linewidth=LINEWIDTH))
    plot.annotate('Low stability', xy=(xpos, 0.5*(max+min)), xytext=(1.4, 42),
                arrowprops=dict(arrowstyle='->', color='black'),
                  fontsize=FONTSIZE)

    # Annotate for NMC
    max = 60.5
    min = 45.5
    xpos = 1.1
    # Make a bidirectional arrow
    HEAD_WIDTH = 0.05
    HEAD_LENGTH = 1
    LINEWIDTH = 1
    FONTSIZE = 16
    plt.arrow(xpos, min, 0, max - min, color='black', head_width=HEAD_WIDTH, head_length=HEAD_LENGTH,
              length_includes_head=True)
    plt.arrow(xpos, max, 0, min - max, color='black', head_width=HEAD_WIDTH, head_length=HEAD_LENGTH,
              length_includes_head=True)
    # Add annotation in the right angle close to the arrow
    # plot.annotate("stability gap", xy=(xpos + 0.03, min - 5), xytext=(xpos + 0.03, min -5 ), rotation=90,
    #               fontsize=FONTSIZE)
    plot.annotate('High stability', xy=(xpos, 0.5*(max+min)), xytext=(1.4, 52),
                arrowprops=dict(arrowstyle='->', color='black'),
                  fontsize=FONTSIZE)

    # Draw dashed lines from 1 to arrow heads
    # plt.axline((1, min), (xpos, min), marker=None, color='black', linestyle='dashed')
    # plt.axline((1, max), (xpos, max), marker=None, color='black', linestyle='dashed')
    plot.add_line(
        matplotlib.lines.Line2D([xpos, 1.0], [min, min], color='black', linestyle='dashed', linewidth=LINEWIDTH))
    plot.add_line(
        matplotlib.lines.Line2D([xpos, 1.0], [max, max], color='black', linestyle='dashed', linewidth=LINEWIDTH))

    # Plot legend
    handles, labels = plot.get_legend_handles_labels()
    handles = [
        handles[labels.index(NAME_FT)],
        handles[labels.index(NAME_NMC_EX)],
        # handles[labels.index(NAME_NMC_FULL)],
    ]
    plot.legend(
        handles=handles,
        labels=labels,
        loc="lower right",
        fontsize=LEGEND_FONTSIZE,
        title=None,
    )

    # Save figure
    plt.tight_layout()
    plt.savefig(str(output_path_png))
    plt.savefig(str(output_path_pdf))


if __name__ == "__main__":
    main()
