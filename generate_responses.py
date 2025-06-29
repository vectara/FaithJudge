import time
import os
import requests
import json
import re
import csv
import argparse
import datetime
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from openai import OpenAI

from google import genai
from google.genai import types, errors
import anthropic
import glob


def load_model(model_id):
    """
    Load or initialize the model/client once, and return a dictionary
    containing everything needed for generation.
    """
    model_data = {"model_id": model_id, "client": None, "mode": None}

    if "openai" in model_id.lower():
        print(f"Loading OpenAI client for '{model_id}'...")
        model_data["client"] = OpenAI()
        model_data["mode"] = "openai"

    elif "google" in model_id.lower() and "gemma" not in model_id.lower():
        print(f"Loading Google GenAI client for '{model_id}'...")
        model_data["mode"] = "google"

    elif "anthropic" in model_id.lower():
        print(f"Loading Anthropic client for '{model_id}'...")
        model_data["client"] = anthropic.Anthropic()
        model_data["mode"] = "anthropic"

    elif "deepseek" in model_id.lower():
        print(f"Loading DeepSeek client for '{model_id}'...")
        deepseek_client = OpenAI(api_key=os.getenv("DeepSeek_API_KEY"), base_url="https://api.deepseek.com")
        model_data["client"] = deepseek_client
        model_data["mode"] = "deepseek"

    elif "grok" in model_id.lower():
        print(f"Loading OpenAI client for '{model_id}'...")
        model_data["client"] = OpenAI(api_key=os.environ.get('XAI_API_KEY'), base_url="https://api.x.ai/v1",)
        model_data["mode"] = "grok"

    elif any(model_provider in model_id.lower() for model_provider in ["meta-llama", "microsoft", "qwen", "gemma", 'thudm', 'mistralai', 'ai21labs']):
        print(f"Loading HuggingFace pipeline for HGF model '{model_id}' locally. This may be large...")
        pipe = pipeline(
            "text-generation",
            model=model_id,
            tokenizer=AutoTokenizer.from_pretrained(model_id),
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
            trust_remote_code=True
        )
        model_data["client"] = pipe
        model_data["mode"] = "hgf"
    
    else:
        print(f"Will use OpenRouter for '{model_id}'...")
        model_data["mode"] = "openrouter"

    return model_data


def generate_response(model_data, system_prompt, user_prompt, max_retries=100):
    """
    Uses the already-loaded model/client in model_data to generate a response.
    If a 429 (RESOURCE_EXHAUSTED) error occurs, we wait and retry up to max_retries times.
    """
    attempt = 0
    model_id = model_data["model_id"]
    mode = model_data["mode"]
    client = model_data["client"]

    while attempt < max_retries:
        try:
            # -------------------------
            # 1) OPENAI
            # -------------------------
            if mode == "openai":
                print(f"Requesting {model_id} via OpenAI API...")
                if "gpt-3.5" in model_id.lower():
                    response = client.chat.completions.create(
                        model=model_id.replace("openai/", ""),
                        messages=[
                            {"role": "developer", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=0.0,
                        max_completion_tokens=4096,
                    )
                elif "gpt" in model_id.lower():
                    response = client.chat.completions.create(
                        model=model_id.replace("openai/", ""),
                        messages=[
                            {"role": "developer", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=0.0,
                        max_completion_tokens=8192,
                    )
                elif "o1" in model_id.lower():
                    response = client.chat.completions.create(
                        model=model_id.replace("openai/", ""),
                        messages=[
                            {"role": "developer", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        max_completion_tokens=16384,
                    )
                elif "o3" in model_id.lower() or 'o4-mini' in model_id.lower():
                    match = re.search(r"(low|medium|high)$", model_id)
                    if match:
                        think_mode = match.group(1)
                    else:
                        think_mode = "medium"
                    
                    base_model_id = model_id
                    base_model_id = base_model_id.replace("high-", "")
                    base_model_id = base_model_id.replace("medium-", "")
                    base_model_id = base_model_id.replace("low-", "")
                    base_model_id = base_model_id.replace("openai/", "")

                    response = client.chat.completions.create(
                        model=base_model_id,
                        reasoning_effort=think_mode,
                        messages=[
                            {"role": "developer", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        max_completion_tokens=16384,
                    )
                else:
                    response = client.chat.completions.create(
                        model=model_id.replace("openai/", ""),
                        messages=[
                            {"role": "developer", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=0.0,
                        max_completion_tokens=8192,
                    )
                return response.choices[0].message.content

            # -------------------------
            # 2) GOOGLE
            # -------------------------
            elif mode == "google":
                print(f"Requesting {model_id} via Google API...")
                if "thinking" in model_id.lower():      
                    prompt = system_prompt + " " + user_prompt
                    genai_client = genai.Client(api_key=os.getenv("GOOGLE_AI_API_KEY"), http_options={"api_version": "v1alpha"})
                    config = {
                        "thinking_config": {"include_thoughts": False},
                        "temperature": 0,
                    }
                    response = genai_client.models.generate_content(
                        model=model_id.lower().split("google/")[-1],
                        contents=prompt,
                        config=config,
                    )
                    if (response.candidates and
                        response.candidates[0].content and
                        response.candidates[0].content.parts and
                        len(response.candidates[0].content.parts) > 0):
                        return response.candidates[0].content.parts[0].text
                    else:
                        print("Warning: No valid text returned by flash-thinking-exp.")
                        return "No content returned by flash-thinking-exp model."
                else:
                    genai_client = genai.Client(api_key=os.getenv("GOOGLE_AI_API_KEY"))
                    response = genai_client.models.generate_content(
                        model=model_id.lower().split("google/")[-1],
                        contents=user_prompt,
                        config=types.GenerateContentConfig(
                            system_instruction=system_prompt,
                            max_output_tokens=8192,
                            temperature=0
                        )
                    )
                    return response.text

            # -------------------------
            # 3) ANTHROPIC
            # -------------------------
            elif mode == "anthropic":
                base_model_id = model_id.replace('anthropic/', '')
                print(f"Requesting {model_id} via Anthropic API...")
                if "think" in model_id.lower():
                    base_model_id = base_model_id.replace('thinking-', '')
                    message = client.messages.create(
                        model=base_model_id,
                        max_tokens=16384,
                        system=system_prompt,
                        thinking={
                            "type": "enabled",
                            "budget_tokens": 8192
                        },
                        messages=[{
                            "role": "user",
                            "content": user_prompt,
                        }]
                    )
                    return message.content[1].text
                else:
                    message = client.messages.create(
                        model=base_model_id,
                        max_tokens=16384,
                        temperature=0,
                        system=system_prompt,
                        messages=[{
                            "role": "user",
                            "content": user_prompt,
                        }]
                    )
                    return message.content[0].text

            # -------------------------
            # 4) DEEPSEEK
            # -------------------------
            elif mode == "deepseek":
                print(f"Requesting {model_id} via DeepSeek API...")
                if "v3" in model_id.lower():
                    response = client.chat.completions.create(
                        model="deepseek-chat",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        max_tokens=8192,
                        temperature=0,
                        stream=False
                    )
                elif "r1" in model_id.lower():
                    response = client.chat.completions.create(
                        model="deepseek-reasoner",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        max_tokens=16384,
                    )
                else:
                    response = client.chat.completions.create(
                        model="deepseek-chat",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        max_tokens=8192,
                        temperature=0
                    )
                return response.choices[0].message.content

            # -------------------------
            # 5) GROK
            # -------------------------
            elif mode =="grok":
                print(f"Requesting {model_id} via OpenAI API...")
                response = client.chat.completions.create(
                        model=model_id.replace("xai/", ""),
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=0.0,
                        max_completion_tokens=8192,)
                
                return response.choices[0].message.content
            
            # -------------------------
            # 6) General HF Pipeline
            # -------------------------
            elif mode == "hgf":
                print(f"Generating with local pipeline for {model_id}...")
                pipe = client  
                messages = [
                    {"role": "system", "content": system_prompt}, 
                    {"role": "user", "content": user_prompt}
                ]
                outputs = pipe(
                    messages,
                    max_new_tokens=8192,
                    do_sample=False,
                    temperature=0
                )
                try:
                    return outputs[0]["generated_text"][-1]["content"]
                except (KeyError, TypeError, IndexError):
                    return str(outputs[0].get("generated_text", "No output?"))
            

            # -------------------------
            # OPENROUTER (fallback)
            # -------------------------
            else:
                print(f"Requesting {model_id} via OpenRouter API...")
                response = requests.post(
                    url="https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {os.environ.get('OPENROUTER_API_KEY')}",
                        "Content-Type": "application/json"
                    },
                    data=json.dumps({
                        "model": model_id,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        "temperature": 0,
                        "max_tokens": 8192,
                    })
                )
                return response.json()["choices"][0]["message"]["content"]

        except errors.ClientError as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                attempt += 1
                wait_time = 60
                if attempt < max_retries:
                    print(f"Hit 429 RESOURCE_EXHAUSTED; waiting {wait_time}s before retry #{attempt}...")
                    time.sleep(wait_time)
                else:
                    print("Max retries exceeded. Returning error message.")
                    return f"ERROR: Quota exhausted after {max_retries} attempts. Could not complete request."
            else:
                raise

        except requests.RequestException as req_error:
            if "429" in str(req_error):
                attempt += 1
                wait_time = 60
                if attempt < max_retries:
                    print(f"Hit 429 rate-limit from openrouter; waiting {wait_time}s before retry #{attempt}...")
                    time.sleep(wait_time)
                else:
                    print("Max retries for openrouter exceeded.")
                    return f"ERROR: Quota exhausted after {max_retries} attempts (openrouter)."
            else:
                raise

    return "ERROR: Unknown condition ended generation loop."


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate responses from an LLM model for different tasks."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/o3-mini",
        help="Model ID, e.g., openai/o3-mini, etc."
    )
    parser.add_argument(
        "--resume",
        type=str,
        default="false",
        help="Whether to resume from the last generated CSV (true/false)."
    )
    return parser.parse_args()



def find_last_csv(out_dir, file_prefix):
    # Look for CSV files matching file_prefix_*.csv in the given directory
    csv_files = glob.glob(os.path.join(out_dir, f"{file_prefix}_*.csv"))
    if not csv_files:
        return None
    csv_files.sort()
    return csv_files[-1]

def count_skip_lines(input_file, resume_source_id, id_field):
    """Count how many lines to skip until we reach the resume_source_id."""
    skip_count = 0
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            skip_count += 1
            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                continue
            if sample.get(id_field) == resume_source_id:
                break
    return skip_count


def main():
    args = parse_args()
    model_id = args.model
    resume_run = args.resume.lower() == "true"

    input_files = ["eval_data/faithbench_summary.jsonl", "eval_data/ragtruth_summary.jsonl", "eval_data/ragtruth_qa.jsonl", "eval_data/ragtruth_data2txt.jsonl"]

    # For output, we create a folder structure based on task and model.
    parts = model_id.split("/")
    if len(parts) == 2:
        organization, model_name = parts
    else:
        organization = parts[0]
        model_name = parts[0]
    out_dir = os.path.join("generated_outputs", organization, model_name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n=== Loading model/client for '{model_id}' ===")
    model_data = load_model(model_id)

    for input_file in input_files:
        
        dataset = input_file.replace("eval_data/", "").split("_")[0]
        task = input_file.replace("eval_data/", "").split("_")[1].replace(".jsonl", "")

        if task == "summary":
            system_prompt = "You must respond based strictly on the information in a provided passage. Do not incorporate any external knowledge or infer any details beyond what is given in the passage."
            base_prompt = "Provide a concise summary of the following passage, covering the core pieces of information described."
            csv_header = ["source_id", "summary"]
            id_field = "source_id"
            text_field = "source"
        elif task == "qa":
            system_prompt = "You must respond based strictly on the information in provided passages. Do not incorporate any external knowledge or infer any details beyond what is given in the passages."
            base_prompt = ("Provide a concise answer to the following question based on the information in the provided passages.")
            csv_header = ["source_id", "qa_response"]
            id_field = "source_id"
            text_field = "source"
        elif task == "data2txt":
            system_prompt = "You must respond based strictly on the information in the provided structured data in the JSON format. Do not incorporate any external knowledge or infer any details beyond what is given in the data."
            base_prompt = ("Write a concise, objective overview of the following local business, based solely on the structured data provided in JSON format. "
                        "You should include important details and cover key information mentioned in the customers' reviews.")
            csv_header = ["source_id", "overview"]
            id_field = "source_id"
            text_field = "source"

        print(f"\nProcessing input file: {input_file}")
        # Determine resume position if needed for this file.
        resume_source_id = None
        csv_path = None
        file_prefix = task  # used for naming CSV files
        if resume_run:
            last_csv = find_last_csv(out_dir, file_prefix)
            if last_csv:
                print(f"Resuming from last CSV file: {last_csv}")
                csv_path = last_csv  # append to existing file
                with open(last_csv, "r", encoding="utf-8") as csvfile:
                    reader = csv.reader(csvfile)
                    header = next(reader, None)
                    rows = list(reader)
                    if rows:
                        last_row = rows[-1]
                        resume_source_id = last_row[0]  # assume first column is the unique id
                        print(f"Last processed {id_field}: {resume_source_id}")
                    else:
                        print("CSV file is empty. Starting from the beginning.")
            else:
                print("No previous CSV found. Starting a new CSV.")
        if not csv_path:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_name = f"{dataset}_{task}_{timestamp}.csv"
            csv_path = os.path.join(out_dir, csv_name)

        # Count total lines in the input file.
        with open(input_file, "r", encoding="utf-8") as f:
            total_samples = sum(1 for _ in f)

        # If resuming and a resume_source_id is found, count how many lines to skip.
        if resume_run and resume_source_id is not None:
            skip_lines = count_skip_lines(input_file, resume_source_id, id_field)
        else:
            skip_lines = 0

        remaining_total = total_samples - skip_lines

        # Open the input file; open the CSV in append mode if resuming.
        with open(input_file, "r", encoding="utf-8") as fin, \
             open(csv_path, "a", encoding="utf-8", newline="") as csvfile:

            writer = csv.writer(csvfile)
            # If the CSV is new, write header.
            if os.stat(csv_path).st_size == 0:
                writer.writerow(csv_header)

            # Skip already processed lines if resuming.
            if resume_run and resume_source_id is not None:
                for _ in range(skip_lines):
                    next(fin, None)

            current_last_source = None  # for duplicate filtering (used in summary/overview)

            # Process each line using tqdm.
            for line in tqdm(
                fin,
                total=remaining_total,
                desc="Processing samples",
                unit="line",
                dynamic_ncols=True,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [elapsed: {elapsed} left: {remaining}]"
            ):
                line = line.strip()
                if not line:
                    continue

                try:
                    sample = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"Error parsing line: {line}\n{e}")
                    continue

                if task == "qa":
                    identifier = sample.get("question_id", sample.get("source_id", "unknown"))
                    question = sample.get(text_field).get("question")
                    passages = sample.get(text_field).get("passages")

                    passages_text = passages.strip()
                    passage_matches = re.findall(r'(passage \d+:)(.*?)(?=passage \d+:|$)', passages_text, flags=re.DOTALL)
                    formatted_passages = []
                    for header, content in passage_matches:
                        formatted_header = header.strip().capitalize()  # "passage 1:" -> "Passage 1:"
                        formatted_content = content.strip()
                        formatted_passages.append(f"{formatted_header}\n\"{formatted_content}\"")
                    passages = "\n\n".join(formatted_passages)
                                            
                    current_prompt = f"{base_prompt}\n\nQuestion: {question}\n\nPassages:\n\n{passages}"
                    row_identifier = identifier

                elif task == "data2txt":
                    # For summary and overview, use the "source" field.
                    identifier = sample.get("source_id", "unknown")
                    content_text = sample.get(text_field, "")
                    # Duplicate-check: skip if this sample's source is the same as the last processed.
                    if current_last_source is not None and content_text == current_last_source:
                        print(f"Skipping sample {identifier} (duplicate source).")
                        continue
                    current_last_source = content_text
                    formatted_source_context = json.dumps(content_text, indent=4, ensure_ascii=False)
                    current_prompt = f"{base_prompt}\n\nJSON Data:\n\"{formatted_source_context}\""
                    row_identifier = identifier
                else:
                    assert(task == "summary")
                    # For summary and overview, use the "source" field.
                    identifier = sample.get("source_id", "unknown")
                    content_text = sample.get(text_field, "")
                    # Duplicate-check: skip if this sample's source is the same as the last processed.
                    if current_last_source is not None and content_text == current_last_source:
                        print(f"Skipping sample {identifier} (duplicate source).")
                        continue
                    current_last_source = content_text
                    current_prompt = f"{base_prompt}\n\nPassage:\n\"{content_text}\""
                    row_identifier = identifier

                # Generate response using the model.
                response_text = generate_response(
                    model_data, 
                    system_prompt,
                    current_prompt
                )

                # Print output for logging.
                print("\n" + "="*60)
                print(f"{id_field.upper()}: {row_identifier}")
                print("="*60)
                print("GENERATED OUTPUT:")
                print(response_text)
                print("="*60 + "\n")

                # Write only the unique id and the generated output to CSV.
                writer.writerow([row_identifier, response_text])

        print(f"\nAll responses appended to: {csv_path}")

if __name__ == "__main__":
    main()
