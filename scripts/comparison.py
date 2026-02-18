# Policy comparison scripts
import matplotlib.pyplot as plt
import numpy as np
import os

total_runs = 20
# ---------------------
# Data: Success and Instruction-Following Rates
# ---------------------

pi05 = {"pick_and_place": {"real_success": 13/20, "real_instruction": 14/20,
                           "wm_success": 13/20, "wm_instruction": 15/20},
        "stacking": {"real_success": 12/20, "real_instruction": 18/20,
                     "wm_success": 3/20, "wm_instruction": 16/20},
        "put_in_drawer": {"real_success": 12/20, "real_instruction": 13/20,
                           "wm_success": 1/20, "wm_instruction": 16/20}}

pi0 = {"pick_and_place": {"real_success": 11/20, "real_instruction": 13/20,
                         "wm_success": 13/20, "wm_instruction": 14/20}, 
         "stacking": {"real_success": 5/19, "real_instruction": 16/19, 
                      "wm_success": 4/19, "wm_instruction": 16/19},
         "put_in_drawer": {"real_success": 0/16, "real_instruction": 0/16,
                           "wm_success": 1/16, "wm_instruction": 12/16}}

pifast = {"pick_and_place": {"real_success": 11/20, "real_instruction": 12/20,
                            "wm_success": 10/20, "wm_instruction": 13/20}, 
          "stacking": {"real_success": 7/20, "real_instruction": 18/20,
                       "wm_success":    4/20, "wm_instruction": 20/20},
          "put_in_drawer": {"real_success": 1/20, "real_instruction": 6/20,
                            "wm_success": 0/20, "wm_instruction": 13/20}}

markers = {"pi05": "o", "pi0": "s", "pifast": "^"}
task_colors = {"pick_and_place": "tab:blue", "stacking": "tab:orange", "put_in_drawer": "tab:green"}

real_success = {
    "pi05_pp": 13/20, 
    "pi05_stack": 12/20, 
    "pi0_stack": 5/19, 
    "pi0_pp": 11/20,
    "pifast_pp": 11/20,
    "pifast_stack": 7/20,
    "pi0_drawer": 0/16,
    "pi05_drawer": 12/20,
    "pifast_drawer": 1/20
}

real_instruction = {
    "pi05_pp": 14/20, 
    "pi05_stack": 18/20,
    "pi0_stack": 16/19, 
    "pi0_pp": 13/20, 
    "pifast_pp": 12/20,
    "pifast_stack": 18/20,
    "pi0_drawer": 0/16,
    "pi05_drawer": 13/20,
    "pifast_drawer": 6/20
}

wm_success = {
    "pi05_pp": 13/20, 
    "pi05_stack": 3/20, 
    "pi0_stack": 4/19, 
    "pi0_pp": 13/20,
    "pifast_pp": 10/20,
    "pifast_stack": 4/20,
    "pi0_drawer": 1/16,
    "pi05_drawer": 1/20,
    "pifast_drawer": 0/20
}

wm_instruction = {
    "pi05_pp": 15/20, 
    "pi05_stack": 16/20,
    "pi0_stack": 16/19, 
    "pi0_pp": 14/20, 
    "pifast_pp": 13/20,
    "pifast_stack": 20/20,
    "pi0_drawer": 12/16,
    "pi05_drawer": 16/20,
    "pifast_drawer": 13/20,
}

correlation_instr = np.corrcoef(
    [real_instruction[k] for k in real_instruction],
    [wm_instruction[k] for k in wm_instruction]
)[0, 1]

correlation_success = np.corrcoef(
    [real_success[k] for k in real_success], 
    [wm_success[k] for k in wm_success]
)[0, 1]

exclude_tasks = []
# ---------------------
# Prepare data
# ---------------------
keys = list(real_success.keys())

real_success_rates = np.array([real_success[k]  for k in keys])
wm_success_rates   = np.array([wm_success[k]  for k in keys])

real_instr_rates   = np.array([real_instruction[k] for k in keys])
wm_instr_rates     = np.array([wm_instruction[k]  for k in keys])

x = np.arange(len(keys))
width = 0.35
figdir = "/n/fs/irom-testing/world_models/Ctrl-World"
# ---------------------
# Helper: Label bar clusters
# ---------------------
def add_cluster_labels(x_positions, labels, y_offset=0.02):
    """Place each dictionary key centered above each pair of bars."""
    for x_pos, label in zip(x_positions, labels):
        plt.text(
            x_pos, 
            1.02,                   # slightly above the top
            label,
            ha='center', va='bottom', fontsize=10, rotation=30
        )

# ----------------------------------------------------------
# Plot 1 — Real Success vs World-Model Success (Scatter)
# ----------------------------------------------------------
plt.figure(figsize=(6, 6))
# plt.scatter(real_success_rates, wm_success_rates, s=80)

for task, rates in pi05.items():
    if task in exclude_tasks:
        continue
    plt.scatter(rates["real_success"], rates["wm_success"], s=100, 
                marker=markers["pi05"], color=task_colors[task], label=f"pi05 - {task}")
for task, rates in pi0.items():
    if task in exclude_tasks:
        continue
    plt.scatter(rates["real_success"], rates["wm_success"], s=100, 
                marker=markers["pi0"], color=task_colors[task], label=f"pi0 - {task}")
for task, rates in pifast.items():
    if task in exclude_tasks:
        continue
    plt.scatter(rates["real_success"], rates["wm_success"], s=100, 
                marker=markers["pifast"], color=task_colors[task], label=f"pifast - {task}")

# for x, y, label in zip(real_success_rates, wm_success_rates, keys):
#     plt.text(x + 0.005, y + 0.005, label, fontsize=10)

plt.xlabel("Real Success Rate")
plt.ylabel("World-Model Success Rate")
plt.title(f"Real vs. World-Model Success (Correlation: {correlation_success:.2f})")
plt.xlim(0, 1.05)
plt.ylim(0, 1.05)
plt.grid(True, linestyle="--", alpha=0.3)
plt.tight_layout()
plt.legend(loc='upper left', fontsize=8)
plt.savefig(os.path.join(figdir, "real_vs_wm_success.png"))

# ---------------------
# Plot 2 — Real vs WM Instruction Following
# ---------------------
plt.figure(figsize=(6, 6))
# plt.scatter(real_instr_rates, wm_instr_rates, s=80)

# for x, y, label in zip(real_instr_rates, wm_instr_rates, keys):
#     plt.text(x + 0.005, y + 0.005, label, fontsize=10)

for task, rates in pi05.items():
    if task in exclude_tasks:
        continue
    plt.scatter(rates["real_instruction"], rates["wm_instruction"], s=100, 
                marker=markers["pi05"], color=task_colors[task], label=f"pi05 - {task}")
for task, rates in pi0.items():
    if task in exclude_tasks:
        continue
    plt.scatter(rates["real_instruction"], rates["wm_instruction"], s=100, 
                marker=markers["pi0"], color=task_colors[task], label=f"pi0 - {task}")
for task, rates in pifast.items():
    if task in exclude_tasks:
        continue
    plt.scatter(rates["real_instruction"], rates["wm_instruction"], s=100, 
                marker=markers["pifast"], color=task_colors[task], label=f"pifast - {task}")

plt.xlabel("Real Instruction-Following Rate")
plt.ylabel("World-Model Instruction-Following Rate")
plt.title(f"Instruction Following: Real vs. World-Model (Correlation: {correlation_instr:.2f})")
plt.xlim(0, 1.05)
plt.ylim(0, 1.05)
plt.grid(True, linestyle="--", alpha=0.3)
plt.tight_layout()
plt.legend(loc='lower right', fontsize=8)
figpath = os.path.join(figdir, "real_vs_wm_instr.png")
plt.savefig(figpath)
