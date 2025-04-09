import argparse
import json
import os
import math
import pandas as pd
import time
import re
import logging
from tqdm import tqdm
from openai import OpenAI
from openai import AzureOpenAI
from prompt_templates import PromptTemplates
from google import genai
from google.genai import types, errors
import requests

def standardize(text, type):
    text = text.strip()
    text = text.replace("LOW INTRODUCTION OF NEW INFORMATION", "SUBTLE INTRODUCTION OF BASELESS INFORMATION")
    text = text.replace("LOW INTRO OF NEW INFO", "SUBTLE INTRODUCTION OF BASELESS INFORMATION")
    text = text.replace("HIGH INTRO OF NEW INFO", "EVIDENT INTRODUCTION OF BASELESS INFORMATION")
    text = text.replace("HIGH INTRODUCTION OF NEW INFORMATION", "EVIDENT INTRODUCTION OF BASELESS INFORMATION")
    text = text.replace("Subtle Baseless Info", "Subtle Introduction of Baseless Information")
    text = text.replace("Evident Baseless Info", "Evident Introduction of Baseless Information")
    text = text.replace('AIGC: ', f'{type}: ')
    text = text.replace('Generative: ', f'{type}: ')
    text = text.replace('Generated: ', f'{type}: ')
    return text


# Setup logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_final_classification(text):
    pattern = re.compile(r"Final classification:\s*(Inconsistent|Consistent|Invalid)", re.IGNORECASE)
    try:
        match = pattern.search(text)
        if match:
            return match.group(1).capitalize()
    except:
        # we haven't found any instances of this thus far
        print("ERROR EXTRACTING CLASSIFICATION FROM JUDGE RESPONSE: ", text)
        quit()
    return "Inconsistent"


TEMPLATE_MAP = {
    "faithbench-summary": PromptTemplates.FAITHBENCH_SUMMARY_BINARY_CLASSIFICATION_WITH_EXAMPLES,
    "ragtruth-summary": PromptTemplates.RAGTRUTH_SUMMARY_BINARY_CLASSIFICATION_WITH_EXAMPLES,
    "ragtruth-qa": PromptTemplates.RAGTRUTH_QA_BINARY_CLASSIFICATION_WITH_EXAMPLES,
    "ragtruth-data2txt": PromptTemplates.RAGTRUTH_DATA2TXT_BINARY_CLASSIFICATION_WITH_EXAMPLES,
}

def load_all_csv_responses(base_dir, model_identifier):
    if '/' not in model_identifier:
        logging.error("model_identifier should be in 'org/model_name' format.")
        return []
    org, model_name = model_identifier.split('/', 1)

    model_path = os.path.join(base_dir, org, model_name)
    if not os.path.exists(model_path):
        logging.error(f"No directory found: {model_path}")
        return []

    csv_files = [f for f in os.listdir(model_path) if f.endswith('.csv')]
    if not csv_files:
        logging.warning(f"No CSV files found under {model_path}")
        return []

    all_responses = []

    pattern = re.compile(r'^(faithbench|ragtruth)_(\w+)_(\d+_\d+)\.csv$')

    for csv_file in csv_files:
        m = pattern.match(csv_file)
        if not m:
            logging.warning(f"Skipping unrecognized CSV file: {csv_file}")
            continue
        dataset, subtask, _ = m.groups()

        task = dataset + "-" + subtask
        
        full_path = os.path.join(model_path, csv_file)

        try:
            df = pd.read_csv(full_path)
        except Exception as e:
            logging.error(f"Failed reading CSV {full_path}: {e}")
            continue

        possible_cols = ['summary', 'qa_response', 'overview']
        found_col = None
        for c in possible_cols:
            if c in df.columns:
                found_col = c
                break

        if not found_col:
            logging.warning(f"No known response column found in {csv_file} "
                            f"({df.columns.tolist()}). Skipping.")
            continue

        for _, row in df.iterrows():
            source_id = str(row.get('source_id', '')).strip()
            candidate_text = row[found_col]
            if candidate_text is None or (isinstance(candidate_text, float) and math.isnan(candidate_text)):
                candidate_text = ""

            candidate_text = candidate_text[:131072]
            candidate_text = re.sub(r'(?s)^.*?\n</think>\n', '', candidate_text)
            
            record = {
                "task": task,
                "file": csv_file,
                "source_id": source_id,
                "candidate_text": candidate_text,
            }
            all_responses.append(record)

    return all_responses


def load_task_data(jsonl_path, task):
    data_map = {}
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)

            if task == "faithbench-summary":
                source_id = str(entry['source_id'])
                source_text = entry['source'].strip()

                if source_id not in data_map:
                    data_map[source_id] = {
                        'source_text': source_text,
                        'annotated_items': []
                    }

                for ann_summary in entry.get('summaries', []):
                    text_candidate = ann_summary['summary'].strip()
                    annotations = ann_summary.get('annotations', [])
                    data_map[source_id]['annotated_items'].append((text_candidate, annotations))
            
            elif task == "ragtruth-summary":
                source_id = str(entry['source_id'])
                source_text = entry['source'].strip()
                if source_id not in data_map:
                    data_map[source_id] = {
                        'source_text': source_text,
                        'annotated_items': []
                    }
                for ann_response in entry.get('responses', []):
                    text_candidate = ann_response['response'].strip()
                    annotations = ann_response.get('labels', [])
                    data_map[source_id]['annotated_items'].append((text_candidate, annotations))

            elif task == "ragtruth-qa":
                source_id = str(entry['source_id'])
                passages_text = entry["source"]["passages"].strip()
                passage_matches = re.findall(r'(passage \d+:)(.*?)(?=passage \d+:|$)', passages_text, flags=re.DOTALL)
                formatted_passages = []
                for header, content in passage_matches:
                    formatted_header = header.strip().capitalize()  # "passage 1:" -> "Passage 1:"
                    formatted_content = content.strip()
                    formatted_passages.append(f"{formatted_header}\n\"{formatted_content}\"")
                passages_text = "\n\n".join(formatted_passages)
                
                question_text = entry["source"]["question"].strip()
                if source_id not in data_map:
                    data_map[source_id] = {
                        'source_question_text': question_text,
                        'source_passages_text': passages_text,
                        'annotated_items': []
                    }
                for ann_response in entry.get('responses', []):
                    text_candidate = ann_response['response'].strip()
                    annotations = ann_response.get('labels', [])
                    data_map[source_id]['annotated_items'].append((text_candidate, annotations))

            elif task == "ragtruth-data2txt":
                source_id = str(entry['source_id'])
                json_business_data = json.dumps(entry["source"], indent=4, ensure_ascii=False)
                if source_id not in data_map:
                    data_map[source_id] = {
                        'source_text': json_business_data,
                        'annotated_items': []
                    }
                for ann_response in entry.get('responses', []):
                    text_candidate = ann_response['response'].strip()
                    annotations = ann_response.get('labels', [])
                    data_map[source_id]['annotated_items'].append((text_candidate, annotations))

            else:
                logging.warning(f"Unknown task type: {task}. Skipping line.")
                continue

    return data_map



def build_prompt(task, data_map, source_id, current_candidate):
    if task not in TEMPLATE_MAP:
        logging.warning(f"No template found for task={task}. Using fallback.")
        return f"No template found for task={task}"

    template = TEMPLATE_MAP[task]

    record = data_map.get(source_id)
    if not record:
        logging.warning(f"source_id={source_id} not found in data_map for task={task}.")
        return "No reference data found."

    annotated_items = record['annotated_items']

    annotated_examples_str = build_annotated_examples(annotated_items, current_candidate, task)

    if task == "faithbench-summary" or task == "ragtruth-summary":
        source_text = record['source_text']
        final_prompt = template.format(
            source_text.strip(),
            annotated_examples_str.rstrip(),
            current_candidate.strip()
        )
        return final_prompt
    
    elif task == "ragtruth-qa":
        question_text = record.get('source_question_text', "")
        passages_text = record.get('source_passages_text', "")
        final_prompt = template.format(
            question_text.strip(),
            passages_text.strip(),
            annotated_examples_str.rstrip(),
            current_candidate.strip()
        )
        return final_prompt
    
    elif task == "ragtruth-data2txt":
        source_text = record['source_text']
        final_prompt = template.format(
            source_text.strip(),
            annotated_examples_str.rstrip(),
            current_candidate.strip()
        )
        return final_prompt

    return "No matching template usage pattern."


def build_annotated_examples(annotated_items, current_candidate, task):
    prompt_examples = ""
    example_idx = 1
    
    for (text_candidate, annotations) in annotated_items:

        if text_candidate.strip() == current_candidate.strip():
            continue

        if task == 'faithbench-summary':

            prompt_examples += f"\tSummary ({example_idx}): \"{text_candidate}\"\n\n"
            example_idx += 1

            for annotator_index, annotator_annotations in enumerate(annotations):
                prompt_examples += f"\t\tAnnotator ({chr(65 + annotator_index)}) Annotations:\n\n"
                if len(annotator_annotations) == 0:
                    prompt_examples += "\t\t\tNo Hallucinated Spans Identified by Annotator\n\n"
                    continue

                for annotation in annotator_annotations:
                    if "source_span" in annotation and annotation["source_span"]:
                        reference_span = annotation["source_span"].replace("\n", " ").strip()
                        prompt_examples += f"\t\t\tReference Span from Source Article: \"{reference_span}\"\n"
                    if "summary_span" in annotation and annotation["summary_span"]:
                        summary_span = annotation["summary_span"].replace("\n", " ").strip()
                        prompt_examples += f"\t\t\tHallucinated Span from Summary: \"{summary_span}\"\n"
                    if annotation.get("label"):
                        labels = [lbl.split(".")[0] for lbl in annotation["label"]]
                        if "Unwanted" in labels:
                            label = "Unwanted"
                        elif "Benign" in labels:
                            label = "Benign"
                        else:
                            label = labels[0]
                        prompt_examples += f"\t\t\tLabel: \"{label}\"\n"
                    if annotation.get("note"):
                        note = annotation["note"].replace("\n", " ").strip()
                        prompt_examples += f"\t\t\tAnnotator Note: \"{note}\"\n"

                    prompt_examples += "\n"

        elif task == 'ragtruth-summary':
            # Show the candidate's summary text
            prompt_examples += f"\tSummary ({example_idx}): \"{text_candidate}\"\n\n"
            example_idx += 1

            prompt_examples += f"\t\tAnnotations:\n\n"

            for annotation in annotations:
                if 'text' in annotation and len(annotation['text']) > 0:
                    summary_span = annotation['text'].replace('\n', ' ').strip()
                    prompt_examples += f"\t\t\tHallucinated Span from Summary: \"{summary_span}\"\n"
                if len(annotation['label_type']) > 0:
                    label = annotation['label_type']
                    label = standardize(label, "Summary")
                    prompt_examples += f"\t\t\tLabel: \"{label}\"\n"
                if  annotation['meta'] and len(annotation['meta']) > 0:
                    note = annotation['meta'].replace('\n', ' ')
                    note = standardize(note, "Summary")
                    prompt_examples += f"\t\t\tAnnotator Note: \"{note}\"\n"
                if annotation['implicit_true']:
                    prompt_examples += f"\t\t\tWhile the hallucinated information may be correct, it is not mentioned in the source.\n"
                prompt_examples += "\n"
                
            if len(annotations) == 0:
                prompt_examples += "\t\t\tNo Hallucinated Spans Identified by Annotator\n\n"

        elif task == 'ragtruth-qa':
            # Show the candidate's summary text
            prompt_examples += f"\tSummary ({example_idx}): \"{text_candidate}\"\n\n"
            example_idx += 1

            prompt_examples += f"\t\tAnnotations:\n\n"

            for annotation in annotations:
                if 'text' in annotation and len(annotation['text']) > 0:
                    response_span = annotation['text'].replace('\n', ' ').strip()
                    prompt_examples += f"\t\t\tHallucinated Span from Response: \"{response_span}\"\n"
                if len(annotation['label_type']) > 0:
                    label = annotation['label_type']
                    label = standardize(label, "Response")
                    prompt_examples += f"\t\t\tLabel: \"{label}\"\n"
                if  annotation['meta'] and len(annotation['meta']) > 0:
                    note = annotation['meta'].replace('\n', ' ')
                    note = standardize(note, "Response")
                    prompt_examples += f"\t\t\tAnnotator Note: \"{note}\"\n"
                if annotation['implicit_true']:
                    prompt_examples += f"\t\t\tWhile the hallucinated information may be correct, it is not mentioned in the source.\n"
                prompt_examples += "\n"
                
            if len(annotations) == 0:
                prompt_examples += "\t\t\tNo Hallucinated Spans Identified by Annotator\n\n"

        elif task == 'ragtruth-data2txt':
            # Show the candidate's summary text
            prompt_examples += f"\tSummary ({example_idx}): \"{text_candidate}\"\n\n"
            example_idx += 1

            prompt_examples += f"\t\tAnnotations:\n\n"

            for annotation in annotations:
                if 'text' in annotation and len(annotation['text']) > 0:
                    overview_span = annotation['text'].replace('\n', ' ').strip()
                    prompt_examples += f"\t\t\tHallucinated Span from Overview: \"{overview_span}\"\n"
                if len(annotation['label_type']) > 0:
                    label = annotation['label_type']
                    label = standardize(label, "Overview")
                    prompt_examples += f"\t\t\tLabel: \"{label}\"\n"
                if  annotation['meta'] and len(annotation['meta']) > 0:
                    note = annotation['meta'].replace('\n', ' ')
                    note = standardize(note, "Overview")
                    prompt_examples += f"\t\t\tAnnotator Note: \"{note}\"\n"
                if annotation['implicit_true']:
                    prompt_examples += f"\t\t\tWhile the hallucinated information may be correct, it is not mentioned in the source.\n"
                if annotation['due_to_null']:
                    prompt_examples += f"\t\t\tThe hallucinated information may have been caused by a null value in the JSON being incorrectly interpreted as a value of false.\n"    
                prompt_examples += "\n"
            
            if len(annotations) == 0:
                prompt_examples += "\t\t\tNo Hallucinated Spans Identified by Annotator\n\n"


    return prompt_examples


def main():
    parser = argparse.ArgumentParser(description="Evaluate CSV model outputs for hallucinations using annotated references.")
    parser.add_argument("--base_dir", default="generated_outputs",
                        help="Top-level directory with <org>/<model_name>/ CSVs.")
    parser.add_argument("--model", required=True,
                        help="Model identifier in the format 'organization/model_name'.")
    parser.add_argument('--output_dir', type=str, default='eval_results', help='Directory to save evaluation results.')
    parser.add_argument("--judge_model", default="o3-mini",
                        help="Which evaluation model to call (e.g. 'o3-mini').")
    parser.add_argument("--use_azure_api", action="store_true", help="Whether to use Azure.")
    args = parser.parse_args()

    # 1) Load the CSV responses
    all_records = load_all_csv_responses(args.base_dir, args.model)
    if not all_records:
        logging.error("No CSV responses found. Exiting.")
        return

    # Group them by "task" (e.g. "faithbench-summary", "ragtruth-summary", etc.).
    task_to_records = {}
    for rec in all_records:
        t = rec["task"]
        task_to_records.setdefault(t, []).append(rec)

    if args.use_azure_api:
        api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
        if not api_key:
            logging.error("Missing AZURE_OPENAI_API_KEY environment variable.")
            return
        endpoint = os.getenv("AZURE_ENDPOINT", "")
        if not endpoint:
            logging.error("Missing AZURE_ENDPOINT environment variable.")
            return
        eval_client = AzureOpenAI(
            azure_endpoint = endpoint,
            api_key=api_key,  
            api_version="2025-02-01-preview"
        )
    elif "gemini" in args.judge_model:
        eval_client = genai.Client(api_key=os.getenv("GOOGLE_AI_API_KEY"))
    elif "llama" in args.judge_model:
        # use openrouter
        eval_client = None
    else:
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            logging.error("Missing OPENAI_API_KEY environment variable.")
            return
        eval_client = OpenAI(api_key=api_key)

    TASK_JSONL_PATHS = {
            "faithbench-summary": "eval_data/faithbench_summary.jsonl",
            "ragtruth-summary":   "eval_data/ragtruth_summary.jsonl",
            "ragtruth-qa":        "eval_data/ragtruth_qa.jsonl",
            "ragtruth-data2txt":  "eval_data/ragtruth_data2txt.jsonl",
        }
    
    for task, record_list in task_to_records.items():
        # Store final results
        results = []
        total, inconsistent_count, invalid_count = 0, 0, 0

        if task not in TASK_JSONL_PATHS:
            logging.warning(f"No annotation file known for task='{task}'. Skipping.")
            continue

        # Load annotated references
        jsonl_path = TASK_JSONL_PATHS[task]
        logging.info(f"Loading reference data for task={task} from {jsonl_path} ...")
        data_map = load_task_data(jsonl_path, task)
        if not data_map:
            logging.warning(f"No reference data loaded from {jsonl_path}. Skipping task={task}.")
            continue

        # Evaluate each record for this task
        for rec in tqdm(record_list, desc=f"Evaluating {task}", unit="item"):
            total += 1
            source_id = rec["source_id"]
            candidate_text = rec["candidate_text"]

            # Build prompt
            prompt = build_prompt(task, data_map, source_id, candidate_text)

            if prompt.startswith("No reference data found") or prompt.startswith("No template"):
                # Mark as invalid
                classification = "Invalid"
                invalid_count += 1
                results.append({
                    "task": task,
                    "source_id": source_id,
                    "candidate_text": candidate_text,
                    "prompt": prompt,
                    "evaluation_response": "(No prompt built)",
                    "classification": classification
                })
                continue

            max_tries = 5
            for attempt in range(max_tries):
                try:
                    if "o3" in args.judge_model:
                        eval_text = eval_client.chat.completions.create(
                            model=args.judge_model,
                            messages=[{
                                "role": "user",
                                "content": prompt
                            }],
                            max_completion_tokens=100000,
                            reasoning_effort="high").choices[0].message.content
                        break
                    elif "gemini" in args.judge_model:
                        eval_text = eval_client.models.generate_content(
                            model=args.judge_model,
                            contents=prompt,
                            config=types.GenerateContentConfig(
                                max_output_tokens=100000,
                                temperature=0
                            )
                        ).text
                        break
                    elif "llama" in args.judge_model:
                        response = requests.post(
                            url="https://openrouter.ai/api/v1/chat/completions",
                            headers={
                                "Authorization": f"Bearer {os.environ.get('OPENROUTER_API_KEY')}",
                                "Content-Type": "application/json"
                            },
                            data=json.dumps({
                                "model": "meta-llama/" + args.judge_model,
                                "messages": [
                                    {"role": "user", "content": prompt}
                                ],
                                "temperature": 0,
                                "max_tokens": 8192,
                            })
                        )
                        eval_text = response.json()["choices"][0]["message"]["content"]
                        break
                except Exception as e:
                    logging.error(f"Eval request error (attempt {attempt+1}): {e}")
                    time.sleep(2)
            else:
                logging.error(f"Max attempts exceeded for source_id={source_id}. Marking as Unknown.")
                classification = "Unknown"
                invalid_count += 1
                results.append({
                    "task": task,
                    "source_id": source_id,
                    "candidate_text": candidate_text,
                    "prompt": prompt,
                    "evaluation_response": "(No response, error)",
                    "classification": classification
                })
                continue

            classification = extract_final_classification(eval_text)
            if classification == "Invalid":
                invalid_count += 1
            elif classification == "Inconsistent":
                inconsistent_count += 1

            # Store result
            results.append({
                "task": task,
                "source_id": source_id,
                "candidate_text": candidate_text,
                "prompt": prompt,
                "evaluation_response": eval_text,
                "classification": classification
            })

        org, model_name = args.model.split('/', 1)
        os.makedirs(os.path.join(args.output_dir, org, model_name), exist_ok=True)
        output_filename = os.path.join(args.output_dir, org, model_name, f"{args.judge_model}_{task}_hallucination_eval.txt")
        
        # Write out results
        with open(output_filename, "w", encoding="utf-8") as outf:
            for r in results:
                outf.write(json.dumps(r, ensure_ascii=False) + "\n")

        # Print stats
        if total > 0:
            hallucination_rate = float(inconsistent_count) / total
            answer_rate = float(total - invalid_count) / total
        else:
            hallucination_rate = 0
            answer_rate = 0

        logging.info(f"\n=== EVALUATION COMPLETE ===")
        logging.info(f"Total: {total}")
        logging.info(f"Inconsistent: {inconsistent_count}")
        logging.info(f"Invalid: {invalid_count}")
        logging.info(f"Hallucination Rate: {hallucination_rate:.2%}")
        logging.info(f"Answer Rate: {answer_rate:.2%}")

if __name__ == "__main__":
    main()