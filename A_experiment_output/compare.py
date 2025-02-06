#!/usr/bin/env python3
"""
compare_all_experiments.py

This script compares results from three experiments:
  1) case1
  2) dirichlet non-iid
  3) case2

For each experiment, we produce two pages in the output PDF:

 Page A: Test Loss vs. NN Pass subplots
   - Top row: FedGM_SASS (red) vs FedGM_SGD (blue)
   - Bottom row: FedGM_SASS (red) vs FedAvg   (green)
   (log-scale y-axis)

 Page B: Server Accuracy vs Global Round
   - Single plot for FedGM_SASS (red), FedGM_SGD (blue), FedAvg (green)
   - A text block in the top-left corner listing each algorithm's final accuracy

All figures are combined into a multi-page PDF:
  /home/local/ASURITE/yzeng88/fedSASS/A_experiment_output/compare_all_experiments.pdf
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.backends.backend_pdf import PdfPages

###############################################################################
# Experiment directories
###############################################################################

# ---------------------------
# Experiment 1: case1
# ---------------------------
fedgm_sass_dir_exp1 = "/home/local/ASURITE/yzeng88/fedSASS/A_experiment_output/fedgm_case1_5clients_50rounds"
fedgm_sgd_dir_exp1  = "/home/local/ASURITE/yzeng88/fedSASS/A_experiment_output/fedgm_sgd_case1_5clients_50rounds"
fedavg_dir_exp1     = "/home/local/ASURITE/yzeng88/fedSASS/A_experiment_output/fedavg_case1_5clients_50rounds"

# ---------------------------
# Experiment 2: dirichlet
# ---------------------------
fedgm_sass_dir_exp2 = "/home/local/ASURITE/yzeng88/fedSASS/A_experiment_output/fedgm_dirichlet_5clients_50rounds"
fedgm_sgd_dir_exp2  = "/home/local/ASURITE/yzeng88/fedSASS/A_experiment_output/fedgm_sgd_dirichlet_5clients_50rounds"
fedavg_dir_exp2     = "/home/local/ASURITE/yzeng88/fedSASS/A_experiment_output/fedavg_dirichlet_5clients_50rounds"

# ---------------------------
# Experiment 3: case2
# ---------------------------
fedgm_sass_dir_exp3 = "/home/local/ASURITE/yzeng88/fedSASS/A_experiment_output/fedgm_case2_5clients_50rounds"
fedgm_sgd_dir_exp3  = "/home/local/ASURITE/yzeng88/fedSASS/A_experiment_output/fedgm_sgd_case2_5clients_50rounds"
fedavg_dir_exp3     = "/home/local/ASURITE/yzeng88/fedSASS/A_experiment_output/fedavg_case2_5clients_50rounds"

###############################################################################
# 1) Reading "loss.txt"
###############################################################################
def read_loss_file(file_path):
    """
    Reads a 'loss.txt' file and returns a dict mapping:
      client_id -> list of (NNPass_Test, TestLoss)

    Expected format (5 columns):
      ClientID  NNPass_Train  TrainLoss  NNPass_Test  TestLoss
    """
    data = {}
    with open(file_path, "r") as f:
        header = f.readline()  # skip the header
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            # columns: client_id, nnpass_train, train_loss, nnpass_test, test_loss
            cid = int(parts[0])
            nn_pass_test = float(parts[3])
            test_loss = float(parts[4])

            if cid not in data:
                data[cid] = []
            data[cid].append((nn_pass_test, test_loss))

    # Sort each client's list by NNPass_Test
    for cid in data:
        data[cid].sort(key=lambda tup: tup[0])
    return data

###############################################################################
# 2) Reading "server_accuracy.txt"
###############################################################################
def read_server_accuracy_file(file_path):
    """
    Expects lines of the form:
      GlobalRound   GlobalAccuracy
    Possibly with a header line. Returns a list of (round_number, accuracy).
    """
    data = []
    with open(file_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        # e.g. "1   92.1234"
        # if the first part is not a digit, skip (it's likely a header)
        if not parts[0].isdigit():
            continue
        rnd = int(parts[0])
        acc = float(parts[1])
        data.append((rnd, acc))

    data.sort(key=lambda t: t[0])
    return data

###############################################################################
# 3) Plot: Test Loss vs. NN Passes
###############################################################################
def create_loss_figure(exp_title, sass_dir, sgd_dir, fedavg_dir):
    """
    2-row subplots:
     - Top row: FedGM_SASS (red) vs FedGM_SGD (blue)
     - Bottom row: FedGM_SASS (red) vs FedAvg (green)
    each subplot is for a client, log-scale y-axis.
    """
    sass_loss_file   = os.path.join(sass_dir,   "loss.txt")
    sgd_loss_file    = os.path.join(sgd_dir,    "loss.txt")
    fedavg_loss_file = os.path.join(fedavg_dir, "loss.txt")

    for fpath in [sass_loss_file, sgd_loss_file, fedavg_loss_file]:
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Could not find file: {fpath}")

    data_sass   = read_loss_file(sass_loss_file)
    data_sgd    = read_loss_file(sgd_loss_file)
    data_fedavg = read_loss_file(fedavg_loss_file)

    all_clients = sorted(list(set(data_sass.keys()) | set(data_sgd.keys()) | set(data_fedavg.keys())))
    n_clients = len(all_clients)

    fig, axes = plt.subplots(nrows=2, ncols=n_clients, figsize=(6*n_clients, 10), squeeze=False)
    fig.subplots_adjust(top=0.90, bottom=0.08, hspace=1.8)

    # Top row: FedGM_SASS vs. FedGM_SGD
    for j, cid in enumerate(all_clients):
        ax = axes[0, j]
        if cid in data_sass:
            x_sass, y_sass = zip(*data_sass[cid])
            ax.plot(x_sass, y_sass, label="FedGM_SASS", color="red", marker="o")
        if cid in data_sgd:
            x_sgd, y_sgd = zip(*data_sgd[cid])
            ax.plot(x_sgd, y_sgd, label="FedGM_SGD", color="blue", marker="s")
        ax.set_title(f"Client {cid}")
        ax.set_xlabel("NN Passes")
        ax.set_ylabel("Test Loss")
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(ticker.LogLocator(base=10, subs=(1.0,)))
        ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext())
        ax.grid(True)
        ax.legend()

    fig.text(0.5, 0.96, f"{exp_title}: FedGM_SASS vs FedGM_SGD", ha="center", fontsize=14)

    # Bottom row: FedGM_SASS vs FedAvg
    for j, cid in enumerate(all_clients):
        ax = axes[1, j]
        if cid in data_sass:
            x_sass, y_sass = zip(*data_sass[cid])
            ax.plot(x_sass, y_sass, label="FedGM_SASS", color="red", marker="o")
        if cid in data_fedavg:
            x_favg, y_favg = zip(*data_fedavg[cid])
            ax.plot(x_favg, y_favg, label="FedAvg", color="green", marker="s")
        ax.set_title(f"Client {cid}")
        ax.set_xlabel("NN Passes")
        ax.set_ylabel("Test Loss")
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(ticker.LogLocator(base=10, subs=(1.0,)))
        ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext())
        ax.grid(True)
        ax.legend()

    fig.text(0.5, 0.44, f"{exp_title}: FedGM_SASS vs FedAvg", ha="center", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    return fig

###############################################################################
# 4) Plot: Server Accuracy vs Global Round + final-acc text
###############################################################################
def create_accuracy_figure(exp_title, sass_dir, sgd_dir, fedavg_dir):
    """
    Single-axes figure that plots server accuracy vs global round for
     FedGM_SASS (red), FedGM_SGD (blue), FedAvg (green).
    We place a text block in the top-left corner showing final accuracies
    for each algorithm.
    """
    sass_acc_file   = os.path.join(sass_dir,   "server_accuracy.txt")
    sgd_acc_file    = os.path.join(sgd_dir,    "server_accuracy.txt")
    fedavg_acc_file = os.path.join(fedavg_dir, "server_accuracy.txt")

    for fpath in [sass_acc_file, sgd_acc_file, fedavg_acc_file]:
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Could not find server_accuracy.txt: {fpath}")

    data_sass   = read_server_accuracy_file(sass_acc_file)
    data_sgd    = read_server_accuracy_file(sgd_acc_file)
    data_fedavg = read_server_accuracy_file(fedavg_acc_file)

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.subplots_adjust(top=0.85)
    
    # We'll store final accuracies in a list of strings to display as text.
    lines_text = []

    # FedGM_SASS
    if data_sass:
        x_sass, y_sass = zip(*data_sass)
        ax.plot(x_sass, y_sass, color='red', marker='o', label="FedGM_SASS")
        final_acc_sass = y_sass[-1]
        lines_text.append(f"FedGM_SASS final acc: {final_acc_sass:.2f}%")

    # FedGM_SGD
    if data_sgd:
        x_sgd, y_sgd = zip(*data_sgd)
        ax.plot(x_sgd, y_sgd, color='blue', marker='s', label="FedGM_SGD")
        final_acc_sgd = y_sgd[-1]
        lines_text.append(f"FedGM_SGD final acc : {final_acc_sgd:.2f}%")

    # FedAvg
    if data_fedavg:
        x_favg, y_favg = zip(*data_fedavg)
        ax.plot(x_favg, y_favg, color='green', marker='^', label="FedAvg")
        final_acc_favg = y_favg[-1]
        lines_text.append(f"FedAvg final acc    : {final_acc_favg:.2f}%")

    # Put a text block in the top-left corner
    if lines_text:
        text_block = "\n".join(lines_text)
        ax.text(0.02, 0.95, text_block,
                transform=ax.transAxes, va='top', ha='left', fontsize=11,
                bbox=dict(facecolor='white', alpha=0.7))

    ax.set_title(f"{exp_title} - Server Accuracy vs Global Round")
    ax.set_xlabel("Global Round")
    ax.set_ylabel("Server Accuracy (%)")
    ax.grid(True)
    ax.legend()
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    return fig

###############################################################################
# 5) MAIN
###############################################################################
def main():
    from matplotlib.backends.backend_pdf import PdfPages
    output_pdf = "/home/local/ASURITE/yzeng88/fedSASS/A_experiment_output/compare_all_experiments.pdf"
    pp = PdfPages(output_pdf)

    # ----------------------------------------------------------------------
    # Experiment 1: case1
    # ----------------------------------------------------------------------
    exp1_title = "Experiment 1: case1"

    # Page A: Test Loss subplots
    fig1 = create_loss_figure(exp_title=exp1_title,
                              sass_dir=fedgm_sass_dir_exp1,
                              sgd_dir=fedgm_sgd_dir_exp1,
                              fedavg_dir=fedavg_dir_exp1)
    pp.savefig(fig1)
    plt.close(fig1)

    # Page B: Accuracy vs global round
    fig1_acc = create_accuracy_figure(exp_title=exp1_title,
                                      sass_dir=fedgm_sass_dir_exp1,
                                      sgd_dir=fedgm_sgd_dir_exp1,
                                      fedavg_dir=fedavg_dir_exp1)
    pp.savefig(fig1_acc)
    plt.close(fig1_acc)

    # ----------------------------------------------------------------------
    # Experiment 2: dirichlet
    # ----------------------------------------------------------------------
    exp2_title = "Experiment 2: dirichlet non-iid"

    # Page A
    fig2 = create_loss_figure(exp_title=exp2_title,
                              sass_dir=fedgm_sass_dir_exp2,
                              sgd_dir=fedgm_sgd_dir_exp2,
                              fedavg_dir=fedavg_dir_exp2)
    pp.savefig(fig2)
    plt.close(fig2)

    # Page B
    fig2_acc = create_accuracy_figure(exp_title=exp2_title,
                                      sass_dir=fedgm_sass_dir_exp2,
                                      sgd_dir=fedgm_sgd_dir_exp2,
                                      fedavg_dir=fedavg_dir_exp2)
    pp.savefig(fig2_acc)
    plt.close(fig2_acc)

    # ----------------------------------------------------------------------
    # Experiment 3: case2
    # ----------------------------------------------------------------------
    exp3_title = "Experiment 3: case2"

    # Page A
    fig3 = create_loss_figure(exp_title=exp3_title,
                              sass_dir=fedgm_sass_dir_exp3,
                              sgd_dir=fedgm_sgd_dir_exp3,
                              fedavg_dir=fedavg_dir_exp3)
    pp.savefig(fig3)
    plt.close(fig3)

    # Page B
    fig3_acc = create_accuracy_figure(exp_title=exp3_title,
                                      sass_dir=fedgm_sass_dir_exp3,
                                      sgd_dir=fedgm_sgd_dir_exp3,
                                      fedavg_dir=fedavg_dir_exp3)
    pp.savefig(fig3_acc)
    plt.close(fig3_acc)

    # ----------------------------------------------------------------------
    # Done, close PDF
    # ----------------------------------------------------------------------
    pp.close()
    print(f"\nCombined PDF saved to: {output_pdf}\n")

if __name__ == "__main__":
    main()
