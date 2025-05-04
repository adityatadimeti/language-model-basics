import os
import pickle
import pandas as pd
import re

# Mapping from d_model to model size
model_size_map = {
    768: "small",
    1024: "medium",
    1280: "large",
    1600: "xl",
    2560: "2.7B",
}

def parse_filename(filename):
    """Extract metadata from filename."""
    match = re.match(r'bench_ctx(\d+)_d(\d+)_ff(\d+)_l(\d+)_h(\d+)\.pkl', filename)
    if match:
        ctx, d_model, d_ff, n_layers, n_heads = map(int, match.groups())
        size = model_size_map.get(d_model, "unknown")
        return {
            "model_size": size,
            "context_length": ctx,
            "d_model": d_model,
            "d_ff": d_ff,
            "num_layers": n_layers,
            "num_heads": n_heads
        }
    else:
        raise ValueError(f"Filename {filename} not recognized!")

def load_all_benchmarks(directory):
    records = []
    for filename in os.listdir(directory):
        if filename.endswith(".pkl"):
            filepath = os.path.join(directory, filename)
            with open(filepath, "rb") as f:
                result = pickle.load(f)
            meta = parse_filename(filename)
            record = {
                **meta,
                "mean_forward": result["mean_forward"],
                "std_forward": result["std_forward"],
                "mean_backward": result["mean_backward"],
                "std_backward": result["std_backward"],
            }
            records.append(record)
    return pd.DataFrame(records)

# Load both sets
#no_warmup_path = "benchmarking_results_no_warmup"
warmup_path = "benchmarking_results_warmup_one"

# Load both sets
#df_no_warmup = load_all_benchmarks(no_warmup_path)
df_warmup    = load_all_benchmarks(warmup_path)

def format_time(mean, std):
    return f"{mean:.5f} ± {std:.5f}"

def df_to_latex(df):
    tmp = df.copy()

    tmp["forward"]  = tmp.apply(lambda row: format_time(row["mean_forward"],  row["std_forward"]),  axis=1)
    tmp["backward"] = tmp.apply(lambda row: format_time(row["mean_backward"], row["std_backward"]), axis=1)

    table = tmp[["model_size", "context_length", "forward", "backward"]]
    table = table.sort_values(by=["model_size", "context_length"])
    return table.to_latex(index=False, escape=False, column_format="|c|c|c|c|")

latex_warmup    = df_to_latex(df_warmup)

#print("% no warmup\n", latex_no_warmup)
print("% with warmup\n",    latex_warmup)