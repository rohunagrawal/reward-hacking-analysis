import json
import os


def plot_scores_across_checkpoints(scores, output_fn: str, y_label: str, title: str, score_name: str):
    """
    scores: dict
        {
            bin_name: {
                baseline: { score_name: value, ... },
                checkpoint_name: { score_name: value, ... },
                ...
            },
            ...
        }
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))

    # Determine universal step ordering from the first bin
    sample_bin = next(iter(scores.values()))
    ordered_ckpts = sorted(sample_bin.keys(), key=lambda x: x)

    # Construct step names using your old logic
    step_names = []
    for name in ordered_ckpts:
        if name == "baseline":
            step_names.append("base")
        elif "_final" in name:
            step_names.append("final")
        else:
            step_names.append(name.split("_checkpoint_")[-1].lstrip("0"))

    # Plot each bin
    for bin_name, ckpt_dict in scores.items():
        bin_scores = []
        for step_name in step_names:
            for key in ckpt_dict.keys():
                if (step_name == "base" and key == "baseline") or (step_name == "final" and "_final" in key) or (step_name not in ["baseline", "final"] and step_name == key.split("_checkpoint_")[-1].lstrip("0")):
                    bin_scores.append(ckpt_dict[key][score_name])
        plt.plot(step_names, bin_scores, marker='o', label=bin_name)

    plt.xlabel("Step")
    plt.ylabel(y_label)
    plt.title(title)
    plt.xticks(rotation=45)
    plt.legend(title="Bins")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_fn)
    plt.close()


def main():
    g_type = "compilability"  # or "llm_judge"
    reward_type = "g"
    model_name = "meta-llama/Llama-3.2-3B"
    out_dir = f"outputs/rl-leetcode/meta-llama/Llama-3.2-3B/{reward_type}"
    bin_output_dict = {
        "Easy": {},
        "Medium": {},
        "Hard": {}
    }

    for difficulty in ["Easy", "Medium", "Hard"]:
        f_g_score_json_fn = os.path.join(out_dir, difficulty, f"avg_heldout_f_g_scores.json")
        with open(f_g_score_json_fn, "r") as f:
            ckpt_to_avg_f_g_vals = json.load(f)
            bin_output_dict[difficulty] = ckpt_to_avg_f_g_vals

    if g_type == "llm_judge":
        bin_name = " w/ reward=g (g=LLM Judge)"
    else:
        if reward_type == "f_g":
            reward_name = "f+g"
        elif reward_type == "100f_g":
            reward_name = "100*f+g"
        else:
            reward_name = reward_type
        bin_name = f" w/ reward={reward_name} (g=Compilability)"

    # Plot scores across checkpoints
    plot_scores_across_checkpoints(
        scores=bin_output_dict,
        output_fn=os.path.join(out_dir, "heldout_g_minus_f_scores.png"),
        y_label="Avg g - f Score",
        title=f"Avg g - f Scores on Dev Set: {model_name}{bin_name}",
        score_name="avg_g_minus_f"
    )
    # Plot scores across checkpoints
    plot_scores_across_checkpoints(
        scores=bin_output_dict,
        output_fn=os.path.join(out_dir, "heldout_f_scores.png"),
        y_label="Avg f Score",
        title=f"Avg f Scores on Dev Set: {model_name}{bin_name}",
        score_name="avg_f"
    )
    # Plot scores across checkpoints
    plot_scores_across_checkpoints(
        scores=bin_output_dict,
        output_fn=os.path.join(out_dir, "heldout_g_scores.png"),
        y_label="Avg g Score",
        title=f"Avg g Scores on Dev Set: {model_name}{bin_name}",
        score_name="avg_g"
    )


if __name__ == "__main__":
    main()