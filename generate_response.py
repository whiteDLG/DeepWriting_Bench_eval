import json
import os
import re
import argparse
from tqdm import tqdm
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import multiprocessing
from multiprocessing import Manager

openai_api_key = "sk-xxx"  # 随便填写，只是为了通过接口参数校验
openai_api_base = "http://localhost:8001/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

def save_output(output, file_name):
    """
    Saves output data to a specified file in JSONL format.
    """
    with open(file_name, 'a', encoding='utf-8') as f:
        for record in output:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

def load_file(file_name):
    """
    Loads JSONL lines from a file into a list of dictionaries.
    """
    if os.path.isfile(file_name):
        with open(file_name, 'r', encoding='utf-8') as f:
            records = [json.loads(line) for line in f]
            return records, len(records)
    return [], 0

def writer(query):
    """
    Function that communicates with OpenAI to get a response for the query.
    """
    resp = client.chat.completions.create(
        model="qwen3-8b-sft",
        messages=[{"role": "user", "content": query}],
        temperature=0.7,
        top_p=0.8,
        extra_body={
            "chat_template_kwargs": {"enable_thinking": False},
            "top_k": 20,
            "min_p": 0.0,
        },
    )
    raw_response = resp.choices[0].message.content
    print(raw_response)
    response = re.sub(r"<think>.*?</think>", "", raw_response, flags=re.DOTALL | re.IGNORECASE,).strip()
    print("==================response====================")
    print(response)
    print("==================response_END====================")

    return response

def process_query(content, existing_count, out_file, lock):
    """
    Process a single query in a separate process.
    """
    data = {"index": content["index"]}
    query = content["query"]
    data["response"] = writer(query)
    
    # Acquire lock before writing to the output file
    with lock:
        save_output([data], out_file)

def process(id_query_map, out_file, num_workers=4):
    """
    Function to process all queries in parallel using multiprocessing.
    """
    records, existing_count = load_file(out_file)
    cnt = existing_count
    contents, input_cnt = load_file(id_query_map)

    # Create a manager for shared objects and a lock to synchronize access to the output file
    manager = Manager()
    lock = manager.Lock()

    with tqdm(total=input_cnt, initial=0, desc=f"Processing {id_query_map.split('/')[-1]}") as pbar:
        # Create a pool of workers
        with multiprocessing.Pool(processes=num_workers) as pool:
            results = []
            for i, content in enumerate(contents):
                if existing_count > 0 and i < existing_count:
                    pbar.update()
                    continue
                
                # Use apply_async to process queries in parallel
                result = pool.apply_async(process_query, (content, existing_count, out_file, lock))
                results.append(result)
                
                # Update the progress bar periodically
                if (i + 1) % 10 == 0 or i == input_cnt - 1:
                    pbar.update()

            # Wait for all processes to finish
            for result in results:
                result.get()

    print(f"CNT: {cnt}")
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process lines from an input file.")
    parser.add_argument("--query_file", default="/data/deepwriting/input/benchmark_all_200.jsonl", 
                        type=str, help="Path to the query file.")
    parser.add_argument("--output_file", default="/data/deepwriting/results/model_response.jsonl", 
                        type=str, help="Path to the output file.")
    parser.add_argument("--num_workers", default=4, type=int, help="Number of worker processes.")
    
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    process(args.query_file, args.output_file, num_workers=args.num_workers)
