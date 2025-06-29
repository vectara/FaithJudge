import os
import json
import re
import pandas as pd
import glob

EVAL_DIR = "eval_results"

EXPECTED_LINES = {
    "faithbench-summary_hallucination_eval.txt": 72,
    "ragtruth-data2txt_hallucination_eval.txt": 150,
    "ragtruth-qa_hallucination_eval.txt": 139,
    "ragtruth-summary_hallucination_eval.txt": 150,
}

TASK_ORDER = [
    "FaithBench-Summary",
    "RagTruth-Summary",
    "RagTruth-QA",
    "RagTruth-Data2Txt",
]

TASK_FILES = {
    "FaithBench-Summary": "faithbench-summary_hallucination_eval.txt",
    "RagTruth-Data2Txt": "ragtruth-data2txt_hallucination_eval.txt",
    "RagTruth-QA": "ragtruth-qa_hallucination_eval.txt",
    "RagTruth-Summary": "ragtruth-summary_hallucination_eval.txt",
}

# Mapping for nicely formatted organization names.
ORG_FORMAT = {
    "meta-llama": "Llama",
    "openai": "OpenAI",
    "ai21labs": "AI21 Labs",
    "mistralai": "Mistral AI",
    "microsoft": "Microsoft",
    "anthropic": "Anthropic",
    "google": "Google"
}

# Dictionary mapping model identifiers ("provider/model") to website URLs.
MODEL_LINKS = {
    "openai/o3-high-2025-04-16": "https://platform.openai.com/docs/models/o3",
    "openai/o3-medium-2025-04-16": "https://platform.openai.com/docs/models/o3",
    "openai/o3-low-2025-04-16": "https://platform.openai.com/docs/models/o3",
    "openai/o3-mini-high-2025-01-31": "https://platform.openai.com/docs/models/o3-mini",
    "openai/o3-mini-medium-2025-01-31": "https://platform.openai.com/docs/models/o3-mini",
    "openai/o3-mini-low-2025-01-31": "https://platform.openai.com/docs/models/o3-mini",
    "openai/o4-mini-high-2025-04-16": "https://platform.openai.com/docs/models/o4-mini",
    "openai/o4-mini-medium-2025-04-16": "https://platform.openai.com/docs/models/o4-mini",
    "openai/o4-mini-low-2025-04-16": "https://platform.openai.com/docs/models/o4-mini",
    "openai/gpt-3.5-turbo-0125": "https://platform.openai.com/docs/models/gpt-3.5-turbo",
    "openai/gpt-4o-2024-11-20": "https://platform.openai.com/docs/models/gpt-4o",
    "openai/gpt-4o-mini-2024-07-18": "https://platform.openai.com/docs/models/gpt-4o-mini",
    "openai/gpt-4.1-2025-04-14": "https://platform.openai.com/docs/models/gpt-4.1",
    "openai/gpt-4.1-mini-2025-04-14": "https://platform.openai.com/docs/models/gpt-4.1-mini",
    "openai/gpt-4.5-preview-2025-02-27": "https://platform.openai.com/docs/models/gpt-4.5-preview",
    "anthropic/claude-3-7-sonnet-thinking-20250219": "https://docs.anthropic.com/en/docs/about-claude/models/all-models",
    "anthropic/claude-3-7-sonnet-20250219": "https://docs.anthropic.com/en/docs/about-claude/models/all-models",
    "anthropic/claude-opus-4-thinking-202505149": "https://docs.anthropic.com/en/docs/about-claude/models/all-models",
    "anthropic/claude-opus-4-202505149": "https://docs.anthropic.com/en/docs/about-claude/models/all-models",
    "anthropic/claude-sonnet-4-thinking-202505149": "https://docs.anthropic.com/en/docs/about-claude/models/all-models",
    "anthropic/claude-sonnet-4-202505149": "https://docs.anthropic.com/en/docs/about-claude/models/all-models",
    "google/gemini-2.0-flash-001": "https://ai.google.dev/gemini-api/docs/models#gemini-2.0-flash",
    "google/gemini-2.5-flash": "https://ai.google.dev/gemini-api/docs/models#gemini-2.5-flash",
    "google/gemini-2.5-pro-exp-03-25": "https://ai.google.dev/gemini-api/docs/models#gemini-2.5-pro-preview-03-25",
    "google/gemini-2.5-pro": "https://ai.google.dev/gemini-api/docs/models#gemini-2.5-pro",
    "xai/grok-3": "https://x.ai/api",
    "meta-llama/llama-4-maverick": "https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E-Instruct",
}

# if the number of parameters cannot be correctly inferred from the model name, use this
MODEL_PARAMS = {
    "microsoft/Phi-4-mini-instruct": "3.8B",
    "microsoft/phi-4": "14B",
    "THUDM/glm-4-9b-chat-hf": "9B",
    "ai21labs/AI21-Jamba-mini-1.6": "12B active / 52B total",
    "meta-llama/llama-4-maverick": "17B active / 109B total"
}

def get_param_count(provider, model):
    """
    Infer the parameter count from the model name using a regex.
    If a pattern like '70B' is found in the model name, return it.
    Otherwise, check the MODEL_PARAMS dict using the key 'provider/model'.
    If still not found, return '?'.
    """
    m = re.search(r'(\d+(?:\.\d+)?B)', model)
    if m:
        return m.group(1)
    model_key = f"{provider}/{model}".strip()
    return MODEL_PARAMS.get(model_key, "?")

def create_model_link(provider, model):
    """
    Returns a Markdown-formatted link for the model.
    Uses the URL from MODEL_LINKS if provided, otherwise defaults to the Hugging Face URL.
    """
    model_key = f"{provider}/{model}"
    if model_key in MODEL_LINKS:
        url = MODEL_LINKS[model_key]
    else:
        url = f"https://huggingface.co/{model_key}"
    return f"[{model}]({url})"

def calculate_hallucination_stats(file_path):
    """
    Reads the given file and returns a tuple: (number of hallucinations, total lines).
    A hallucination is counted if the 'classification' is either "Inconsistent" or "Invalid".
    """
    with open(file_path, 'r') as f:
        lines = [json.loads(line.strip()) for line in f if line.strip()]
    hallucinations = sum(1 for line in lines if line['classification'] in ["Inconsistent", "Invalid"])
    return hallucinations, len(lines)

def process_models(eval_dir):
    data = []
    warnings = []

    for provider in os.listdir(eval_dir):
        provider_path = os.path.join(eval_dir, provider)
        if not os.path.isdir(provider_path):
            continue

        # Format organization names nicely using ORG_FORMAT mapping
        formatted_provider = ORG_FORMAT.get(provider.lower(), provider)

        for model in os.listdir(provider_path):
            model_path = os.path.join(provider_path, model)
            if not os.path.isdir(model_path):
                continue

            entry = {
                "Model": create_model_link(provider, model),
                "Organization": formatted_provider,
                "# Parameters": get_param_count(provider, model)
            }

            valid = True
            total_hallucinations = 0
            total_lines = 0

            for task_name in TASK_ORDER:
                suffix = TASK_FILES[task_name]
                matches = glob.glob(os.path.join(model_path, f"*{suffix}"))
                if not matches:
                    warnings.append(f"Missing file: *{suffix}")
                    valid = False
                    break
                file_path = matches[0] 

                if not os.path.exists(file_path):
                    warnings.append(f"Missing file: {file_path}")
                    valid = False
                    break

                count, num_lines = calculate_hallucination_stats(file_path)
                if num_lines != EXPECTED_LINES[suffix]:
                    warnings.append(
                        f"Line count mismatch in {file_path} (expected {EXPECTED_LINES[suffix]}, got {num_lines})"
                    )
                    valid = False
                    break

                percentage = (count / num_lines * 100) if num_lines else 0
                entry[task_name] = f"{percentage:.2f}% ({count}/{num_lines})"

                total_hallucinations += count
                total_lines += num_lines

            if valid and total_lines > 0:
                overall_rate = total_hallucinations / total_lines * 100
                entry["Overall Hallucination Rate"] = f"{overall_rate:.2f}%"
                data.append(entry)
            else:
                print(f"WARNING: Skipping {provider}/{model} due to validation issues.")

    df = pd.DataFrame(data)
    
    df["Numeric Overall Rate"] = df["Overall Hallucination Rate"].str.replace("%", "").astype(float)
    
    # Sort by the numeric value of the overall hallucination rate
    df.sort_values(by="Numeric Overall Rate", inplace=True)
    
    # Reset index and add a Rank column at the left-most position
    df.reset_index(drop=True, inplace=True)
    df.insert(0, "Rank", df.index + 1)

    # Drop the temporary numeric column used for sorting
    df.drop(columns=["Numeric Overall Rate"], inplace=True)

    DATASET_COLUMN_NAMES = {
        "FaithBench-Summary": "Faithbench (Summarization)",
        "RagTruth-Summary": "RagTruth (Summarization)",
        "RagTruth-QA": "RagTruth (Question-Answering)",
        "RagTruth-Data2Txt": "RagTruth (Data-to-Text Writing)"
    }
    
    # Rename the dataset columns in the DataFrame.
    df.rename(columns=DATASET_COLUMN_NAMES, inplace=True)

    # Update columns order to place Organization after Model.
    new_columns_order = ["Rank", "Model", "Organization", "# Parameters", "Overall Hallucination Rate"] \
                        + [DATASET_COLUMN_NAMES[task] for task in TASK_ORDER]
    return df[new_columns_order], warnings

def update_readme_with_table(markdown_table, readme_file="README.md"):
    """
    Updates the README.md by replacing the content between the markers
    <!-- TABLE START --> and <!-- TABLE END --> with the new markdown table.
    If the markers are not found, the table is appended to the end of the file within these markers.
    """
    marker_start = "<!-- TABLE START -->"
    marker_end = "<!-- TABLE END -->"

    # Read the existing README.md content
    try:
        with open(readme_file, "r") as f:
            contents = f.read()
    except FileNotFoundError:
        # If README.md doesn't exist, start with an empty content
        contents = ""
    
    # Check if the markers are present in the file
    if marker_start in contents and marker_end in contents:
        # Build the replacement text including the markers and new table
        replacement = f"{marker_start}\n{markdown_table}\n{marker_end}"
        # Replace everything between the markers using regex
        pattern = re.compile(re.escape(marker_start) + ".*?" + re.escape(marker_end), re.DOTALL)
        updated_contents = pattern.sub(replacement, contents)
    else:
        # If markers are not present, append the table with markers at the end of the file
        updated_contents = contents + f"\n\n{marker_start}\n{markdown_table}\n{marker_end}\n"

    with open(readme_file, "w") as f:
        f.write(updated_contents)

def generate_markdown_table(df):
    """Converts the DataFrame to a markdown table."""
    return df.to_markdown(index=False)

if __name__ == "__main__":
    pd.set_option('display.max_colwidth', 1000)

    df, warnings = process_models(EVAL_DIR)

    if warnings:
        print("Warnings:")
        for warn in warnings:
            print(f" - {warn}")
    else:
        print("No warnings, all files validated successfully.")

    # Generate the markdown table from the DataFrame
    markdown_table = generate_markdown_table(df)
    
    # Update the existing README.md file with the generated markdown table
    update_readme_with_table(markdown_table)
    print("Updated README.md with the new markdown table.")
