import argparse
import json
import os

from matplotlib import pyplot as plt
import numpy as np
import seaborn
from sklearn import linear_model

seaborn.set()
seaborn.set_style("whitegrid")


def make_inference_latency_plot(result_dir, max_elapsed_time=2000):

    token_counts = []  # x
    inference_latencies = []  # y
    model_names = []  # Legend

    linear_regr_models = []  # Linear Regression for expect token_counts (SQuAD's context: 50 ~ 700 tokens)

    for file_path in os.listdir(result_dir):
        result_path = os.path.join(result_dir, file_path)
        if not result_path.endswith(".json"):
            continue

        with open(result_path, "r") as f:
            model_name = os.path.basename(result_path).replace(".json", "")
            model_names.append(model_name)

            result = json.load(f)
            token_count = [r["token_count"] for r in result["tensor_to_predicts"]]
            inference_latency = [r["elapsed_time"] for r in result["tensor_to_predicts"]]

            token_counts.append(token_count)
            inference_latencies.append(inference_latency)

            # Create linear regression for predict token_counts
            regr = linear_model.LinearRegression()
            regr.fit(
                np.array(inference_latency).reshape(-1, 1), np.array(token_count).reshape(-1, 1)
            )
            linear_regr_models.append(regr)

    f_name = f"inference_latency_chart-{max_elapsed_time}.png"
    title = "Inference Latency"

    zipped_data = list(zip(model_names, token_counts, inference_latencies, linear_regr_models))
    zipped_data.sort()

    # get maximum token count
    for zipped in zipped_data:
        model_name, token_counts, inference_latencies, linear_regr_model = zipped

        max_token_count = 0
        for token_count, inference_latency in zip(token_counts, inference_latencies):
            if inference_latency <= max_elapsed_time and max_token_count < token_count:
                max_token_count = token_count

        token_logs = f"model_name: {model_name} | "
        token_logs += f"max_token_count: {max_token_count} "
        token_logs += f"(predict: {int(linear_regr_model.predict(np.array(max_elapsed_time).reshape(-1, 1)))})"
        token_logs += f" / {max_elapsed_time} mills"
        print(token_logs)

    model_names, token_counts, inference_latencies, linear_regr_models = zip(*zipped_data)

    make_scatter(
        token_counts,
        inference_latencies,
        f_name,
        linear_regr_models=linear_regr_models,
        alpha=0.2,
        size=[18, 10],
        s=20,
        legends=model_names,
        x_min=0,
        x_max=800,
        y_min=0,
        y_max=max_elapsed_time,
        x_label="Tokens",
        y_label="1-example Latency (milliseconds)",
        title=title,
        markerscale=5,
    )


def make_summary_plot(result_dir, max_elapsed_time=100):

    max_token_counts = []  # x
    f1_scores = []  # y
    model_names = []  # Legend

    for file_path in os.listdir(result_dir):
        result_path = os.path.join(result_dir, file_path)
        if not result_path.endswith(".json"):
            continue

        with open(result_path, "r") as f:
            model_name = os.path.basename(result_path).replace(".json", "")

            result = json.load(f)

            model_names.append(model_name + "_cpu")
            model_names.append(model_name + "_gpu")

            f1_scores.append([result["metrics"]["best"]["valid/f1"]])
            f1_scores.append([result["metrics"]["best"]["valid/f1"]])

            max_token_counts.append(
                [result["inferency_latency"]["cpu"]["max_token_count"][str(max_elapsed_time)]]
            )
            max_token_counts.append(
                [result["inferency_latency"]["gpu"]["max_token_count"][str(max_elapsed_time)]]
            )

    f_name = f"summary.png"
    title = "Model Summary"

    zipped_data = list(zip(model_names, f1_scores, max_token_counts))
    zipped_data.sort()

    model_names, f1_scores, max_token_counts = zip(*zipped_data)

    latency_min, latency_max = 0, 1000
    f1_min, f1_max = 60, 80

    make_scatter(
        max_token_counts,
        f1_scores,
        f_name,
        is_env_with_color=True,
        size=[18, 10],
        legends=model_names,
        s=400,
        alpha=1,
        y_min=60,
        y_max=80,
        x_min=0,
        x_max=700,
        x_ticks=list(range(latency_min, latency_max + 1, 100)),
        y_ticks=list(range(f1_min, f1_max + 1, 5)),
        x_label=f"Maximum token count ({max_elapsed_time} milliseconds)",
        y_label="F1 Score",
        title=title,
    )


def make_scatter(
    x,
    y,
    f_name,
    is_env_with_color=False,
    linear_regr_models=None,
    size=[10, 14],
    title=None,
    legends=None,
    s=10,
    alpha=0.6,
    markerscale=1,
    x_min=None,
    y_min=None,
    x_max=None,
    y_max=None,
    x_label=None,
    y_label=None,
    x_ticks=None,
    y_ticks=None,
):
    fig = plt.figure(figsize=(size[0], size[1]))

    markers = ["o", "*", "v", "^", "<", ">", "8", "s", "p", "h", "H", "D", "d", "P", "X"]

    if title is not None:
        plt.title(title, fontsize=32)
    if x_label is not None:
        plt.xlabel(x_label, fontsize=22)
    if y_label is not None:
        plt.ylabel(y_label, fontsize=22)
    if x_min is not None or x_max is not None:
        plt.xlim(xmin=x_min, xmax=x_max)
    if y_min is not None or y_max is not None:
        plt.ylim(ymin=y_min, ymax=y_max)
    if x_ticks is not None:
        plt.xticks(x_ticks, x_ticks, fontsize=18)
    else:
        plt.xticks(fontsize=18)
    if y_ticks is not None:
        plt.yticks(y_ticks, y_ticks, fontsize=18)
    else:
        plt.yticks(fontsize=18)

    if isinstance(x[0], list) and isinstance(y[0], list):
        for index, (x_item, y_item) in enumerate(zip(x, y)):
            if is_env_with_color:
                i = int(index / 2)
                if index % 2 == 0:
                    plt.scatter(x_item, y_item, s=s, c="b", marker=markers[i], alpha=alpha)
                else:
                    plt.scatter(x_item, y_item, s=s, c="g", marker=markers[i], alpha=alpha)
            else:
                if index % 2 == 0:
                    plt.scatter(x_item, y_item, s=s, marker="o", alpha=alpha)
                else:
                    plt.scatter(x_item, y_item, s=s, marker="^", alpha=alpha)

    else:
        plt.scatter(x, y, s=s, alpha=alpha)

    if linear_regr_models is not None:
        ys = np.arange(y_min, y_max)
        for model in linear_regr_models:
            xs = [int(model.predict(np.array(y).reshape(-1, 1))) for y in ys]
            plt.plot(xs, ys)

    if legends is not None:
        plt.legend(
            legends,
            fontsize=24,
            fancybox=True,
            shadow=True,
            loc=(1.04, 0.3),
            markerscale=markerscale,
        )

    plt.savefig(f_name, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "plot_type", type=str, default="inference", help="Plot type [inference|summary]"
    )
    parser.add_argument(
        "--result_dir", type=str, default="inference_result", help="SQuAD official json file path"
    )
    parser.add_argument(
        "--max_latency",
        type=int,
        default=2000,
        help="The number of maximum latency time. (milliseconds)",
    )

    config = parser.parse_args()

    if config.plot_type == "inference":
        make_inference_latency_plot(config.result_dir, max_elapsed_time=config.max_latency)
    elif config.plot_type == "summary":
        make_summary_plot(config.result_dir, max_elapsed_time=config.max_latency)
    else:
        raise ValueError(f"not supported plot_type: {config.plot_type}")

    print(f"Complete make {config.plot_type} plot")
