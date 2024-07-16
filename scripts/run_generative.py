# Copyright 2023 AllenAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# run a generative RM. For now, this requires openai and anthropic to be installed
# Examples:
# python scripts/run_generative.py --model gpt-3.5-turbo
# python scripts/run_generative.py --model=claude-3-haiku-20240307

# note: for none API models, this script uses vllm
# pip install vllm

import argparse
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import numpy as np
from fastchat.conversation import get_conv_template
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from collections import Counter
from rewardbench import load_eval_dataset, save_to_hub
from rewardbench.constants import EXAMPLE_COUNTS, SUBSET_MAPPING
from rewardbench.generative import (
    ANTHROPIC_MODEL_LIST,
    API_MODEL_LIST,
    GEMINI_MODEL_LIST,
    OPENAI_MODEL_LIST,
    format_judge_answers,
    process_judgement,
    run_judge_pair,
)
from rewardbench.utils import calculate_scores_per_section
from collections import Counter
from datasets import Dataset

# get token from HF_TOKEN env variable, but if it doesn't exist pass none
HF_TOKEN = os.getenv("HF_TOKEN", None)
# this is necessary to automatically log in when running this script in docker/batch beaker jobs
if HF_TOKEN is not None:
    from huggingface_hub._login import _login

    _login(token=HF_TOKEN, add_to_git_credential=False)

def fix_seed(seed: int = 0):
    np.random.seed(seed)
    # random.seed(seed)

def get_judgement(example_id, batch, args):
    mult_turn = True if len(batch["text_chosen"]) > 2 else False
    prompt = batch["text_chosen"][0]["content"]
    answer_a = batch["text_chosen"]
    answer_b = batch["text_rejected"]

    # shuffle a and b randomly for position bias
    is_shuffled = np.random.rand() > 0.5
    if is_shuffled:
        answer_a, answer_b = answer_b, answer_a

    if len(batch["text_chosen"]) <= 4:  # set up only for 1 or 2 turns
        answer = {'A': answer_a, 'B': answer_b}
        winner, request, judgement = run_judge_pair(
            prompt, answer, args.model, multi_turn=mult_turn, 
            model_modifier=args.model_modifier, use_moa=args.moa,
            evaluation_style=args.evaluation_style, debug=args.debug, temperature=args.temperature
        )

        # handle voting for PoLL
        if isinstance(winner, list) and not args.moa:
            if args.debug:
                print(winner)
            winner = max(set(winner), key=winner.count)
        
        # Save detailed results
        detailed_results = save_detailed_results(example_id, prompt, answer_a, answer_b, winner, judgement, args, is_shuffled)

        # Process the result
        processed_result = process_json_result(detailed_results, args.evaluation_style)
        
        return processed_result, detailed_results
    else:
        return 0.5, {"final_preference": 0.5, "gold_preference": 0.5, "example": {"id": example_id}}


def process_result(winner, winner_text, loser_text):
    if winner == winner_text:
        return 1
    elif winner == loser_text:
        return 0
    else:  # if "error" or tie
        return 0.5  # effectively a tie

def process_json_result(json_data, evaluation_style="binary"):
    if 'aggregator' in json_data and json_data['aggregator']:
        # MoA case
        aggregator_judgment = json_data['aggregator']['output']
    elif 'reference_models' in json_data and json_data['reference_models']:
        # Single model case
        aggregator_judgment = json_data['reference_models'][0]['output']
    else:
        # If neither aggregator nor reference_models are present, return an error
        return 0.5  # Treat as a tie/error case

    winner = process_judgement(aggregator_judgment, evaluation_style=evaluation_style)
    
    # Determine winner_text and loser_text based on gold_preference
    gold_preference = json_data.get('gold_preference', 0.5)  # Default to 0.5 if not present
    if gold_preference == 1:
        winner_text = "A"
        loser_text = "B"
    elif gold_preference == 0:
        winner_text = "B"
        loser_text = "A"
    else:
        # If gold_preference is 0.5 or invalid, treat as a tie
        return 0.5

    return process_result(winner, winner_text, loser_text)

def load_existing_results(args, folder_name, evaluation_style=None):
    if evaluation_style:
        output_folder = os.path.join(args.output_dir, folder_name, evaluation_style)
    else:
        output_folder = os.path.join(args.output_dir, folder_name)
    
    existing_results = {}
    if os.path.exists(output_folder):
        json_files = [f for f in os.listdir(output_folder) if f.endswith("_results.json")]
        for file in json_files:
            try:
                with open(os.path.join(output_folder, file), 'r') as f:
                    data = json.load(f)
                    example_id = int(file.split("_")[0])
                    existing_results[example_id] = process_json_result(data, args.evaluation_style)
            except json.JSONDecodeError:
                print(f"Error decoding JSON in file: {file}")
            except Exception as e:
                print(f"Error processing file {file}: {str(e)}")
    return existing_results

def update_progress_bar(done, total):
    progress = int(50 * done / total)
    sys.stdout.write("\r[{}{}] {}/{}".format("#" * progress, "." * (50 - progress), done, total))
    sys.stdout.flush()


def get_args():
    """
    Parse arguments strings model and chat_template
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        nargs="+",  # allow list of models (ensemble)
        required=False,
        help="name of OpenAI model to use (TODO add more providers/models)",
    )
    parser.add_argument("--chat_template", type=str, default=None, help="fastchat chat template (optional)")
    parser.add_argument(
        "--trust_remote_code", action="store_true", default=False, help="directly load model instead of pipeline"
    )
    parser.add_argument("--num_gpus", type=int, default=1, help="number of gpus to use, for multi-node vllm")
    parser.add_argument("--do_not_save", action="store_true", help="do not save results to hub (for debugging)")
    parser.add_argument(
        "--pref_sets", action="store_true", help="run on common preference sets instead of our custom eval set"
    )
    parser.add_argument(
        "--debug", action="store_true", help="run on common preference sets instead of our custom eval set"
    )
    parser.add_argument(
        "--subset", action="store_true", help="run only on a subset of the data for debugging purposes"
    )
    parser.add_argument(
        "--num_threads", type=int, default=10, help="number of threads to use for parallel processing of examples"
    )
    parser.add_argument(
        "--disable_beaker_save", action="store_true", help="disable saving the main results in a file for AI2 Beaker"
    )
    parser.add_argument(
        "--force_local", action="store_true", default=False, help="force local run, even if model is on Together API"
    )
    parser.add_argument(
        "--temperature", type=float, default=0, help="temperature for sampling (0 for greedy decoding)"
    )   
    parser.add_argument(
        "--moa",
        action="store_true",
        help="Use Mixture of Agents approach",
    )
    parser.add_argument(
        "--reference_models",
        nargs="+",
        default=["microsoft/WizardLM-2-8x22B", "Qwen/Qwen1.5-110B-Chat", "Qwen/Qwen2-72B-Instruct", "meta-llama/Llama-3-70b-chat-hf", "mistralai/Mixtral-8x22B-Instruct-v0.1", "databricks/dbrx-instruct"],
        help="List of reference models for MoA",
    )
    parser.add_argument(
        "--aggregator_model",
        type=str,
        default="Qwen/Qwen2-72B-Instruct",
        help="Aggregator model for MoA",
    )
    parser.add_argument(
        "--evaluation_style",
        type=str,
        default="binary",
        choices=["binary", "score"],
        help="Evaluation style: binary (A/B) or score-based (1-5)",
    )
    parser.add_argument(
        "--single_proposer", action="store_true", help="Use single proposer for MoA",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/usr/project/xtmp/rx55/projects/moa-eval/results/rewardBench/",
        help="Base output directory for saving results"
    )
    args = parser.parse_args()
    
    args.do_not_save = True
    args.force_local = True
    #args.debug = True # False or True
    #args.moa = True # False or True
    #args.subset = True # False or True
    # args.single_proposer = True

    args.num_gpus = 2
    #args.num_threads = 1
    args.reference_models =["microsoft/WizardLM-2-8x22B", "Qwen/Qwen1.5-110B-Chat", "Qwen/Qwen2-72B-Instruct", "meta-llama/Llama-3-70b-chat-hf", "mistralai/Mixtral-8x22B-Instruct-v0.1", "databricks/dbrx-instruct"]
    # args.reference_models =["microsoft/WizardLM-2-8x22B", "Qwen/Qwen1.5-110B-Chat"]
    #args.aggregator_model = "Qwen/Qwen2-72B-Instruct"
    
    args.reference_models =["microsoft/WizardLM-2-8x22B", "Qwen/Qwen1.5-110B-Chat", "Qwen/Qwen2-72B-Instruct", "meta-llama/Llama-3-70b-chat-hf", "mistralai/Mixtral-8x22B-Instruct-v0.1", "databricks/dbrx-instruct"]
    args.aggregator_model = "Qwen/Qwen1.5-110B-Chat"
    
    
    if args.debug:
        args.reference_models =["Qwen/Qwen1.5-0.5B-Chat"]
        args.aggregator_model = "Qwen/Qwen1.5-0.5B-Chat"

    if args.single_proposer:
        args.reference_models = ["meta-llama/Llama-3-70b-chat-hf","meta-llama/Llama-3-70b-chat-hf","meta-llama/Llama-3-70b-chat-hf","meta-llama/Llama-3-70b-chat-hf","meta-llama/Llama-3-70b-chat-hf","meta-llama/Llama-3-70b-chat-hf"]
        args.aggregator_model = "Qwen/Qwen1.5-110B-Chat"
        args.temperature = 1


    args.model = "meta-llama/Llama-3-70b-chat-hf"
    args.model = "Qwen/Qwen2-72B-Instruct"
    args.model = "Qwen/Qwen1.5-110B-Chat"

    #args.model = "meta-llama/Meta-Llama-3-8B-Instruct"
    #args.model = "meta-llama/Llama-3-8b-chat-hf"
     
    
    return args




def main():
    
    fix_seed(0)
    args = get_args()
    ###############
    # Setup logging
    ###############
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = logging.INFO
    logger.setLevel(log_level)

    if args.moa:
        model_type = "Generative RM MoA"
        args.model = args.reference_models + [args.aggregator_model]
        logger.info(f"Running reward models using MoA with reference models {args.reference_models} and aggregator model {args.aggregator_model}")
    elif isinstance(args.model, list) and len(args.model) > 1:
        model_type = "Generative RM PoLL"
        assert len(args.model) % 2 == 1, "Number of models for PoLL must be odd"
    else:
        model_type = "Generative RM"
        logger.info(f"Running reward model on {args.model} with chat template {args.chat_template}")

        if isinstance(args.model, list) and len(args.model) == 1:
            args.model = args.model[0]

    is_api_models = isinstance(args.model, list) or args.model in API_MODEL_LIST or not args.force_local

    # if model isn't API, load via vllm
    if not is_api_models:
        # load model
        model = LLM(args.model, trust_remote_code=args.trust_remote_code, tensor_parallel_size=args.num_gpus)
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        if "Llama-3" in args.model or "llama3-8b" in args.model:
            stop_token_ids = [128009]
        else:
            stop_token_ids = []

        sampling_params = SamplingParams(
            n=1,
            temperature=0,
            top_p=1,
            max_tokens=1024,
            stop_token_ids=stop_token_ids,
        )

    # handle off-case models
    is_prometheus = False  # handles output tokens differently (less flexible)
    # use different prompt for prometheus/gemini models
    if "prometheus" in args.model:
        model_modifier = "prometheus"
        is_prometheus = True
    elif "gemini" in args.model:
        model_modifier = "gemini"
    else:
        model_modifier = None

    ############################
    # Load dataset
    ############################
    logger.info("*** Load dataset ***")
    logger.info(f"*** Evaluation style: {args.evaluation_style} ***")


    dataset, subsets = load_eval_dataset(
        core_set=not args.pref_sets,
        conv=get_conv_template("raw"),  # not used in this script (handled later)
        custom_dialogue_formatting=True,  # handle formatting later
        tokenizer=None,
        logger=logger,
        keep_columns=["text_chosen", "text_rejected", "id", "subset"],
        max_turns=4,
    )

    # copy id for saving, then remove
    ids = dataset["id"]

    # Debug: select a few examples from each subset
    if args.subset:
        debug_size = 25
        debug_indices = []
        
        unique_subsets = set(subsets)
        for subset in unique_subsets:
            subset_indices = [i for i, s in enumerate(subsets) if s == subset]
            selected_indices = subset_indices[:debug_size]
            debug_indices.extend(selected_indices)
        
        # Use the select method to create a new dataset with the debug indices
        dataset = dataset.select(debug_indices)
        subsets = [subsets[i] for i in debug_indices]
        ids = [ids[i] for i in debug_indices]
        logger.info(f"Dataset size after selection: {len(dataset)}")
        logger.info(f"Number of examples per subset: {Counter(subsets)}")

    if args.debug:
        # debug_size = 1
        # dataset = dataset.select(range(debug_size))
        # subsets = subsets[:debug_size]
        # ids = ids[:debug_size]
        debug_size = 1
        debug_indices = []
        
        unique_subsets = set(subsets)
        for subset in unique_subsets:
            subset_indices = [i for i, s in enumerate(subsets) if s == subset]
            selected_indices = subset_indices[:debug_size]
            debug_indices.extend(selected_indices)
        
        # Use the select method to create a new dataset with the debug indices
        dataset = dataset.select(debug_indices)
        subsets = [subsets[i] for i in debug_indices]
        ids = [ids[i] for i in debug_indices]
    
        logger.info(f"Dataset size after debug selection: {len(dataset)}")
        
    if is_api_models:
        ############################
        # Run inference via API
        ############################
        def update_progress_bar(done, total):
            # Simple text-based progress bar
            progress = int(50 * done / total)  # Calculate progress (50 chars width)
            sys.stdout.write("\r[{}{}] {}/{}".format("#" * progress, "." * (50 - progress), done, total))
            sys.stdout.flush()
            
        def save_detailed_results(example_id, prompt, answer_a, answer_b, winner, judgement, args, is_shuffled):
            detailed_results = {
                "example": {
                    "id": example_id,
                    "user_query": prompt,
                    "response_a": answer_a[1]["content"],
                    "response_b": answer_b[1]["content"]
                },
                "reference_models": [],
                "aggregator": None,
                "final_preference": 1 if winner == "A" else (0 if winner == "B" else 0.5),
                "gold_preference": 1 if not is_shuffled else 0
            }

            if args.moa:
                reference_judgments, aggregator_judgment = judgement
                for model, judgment in zip(args.reference_models, reference_judgments):
                    if args.evaluation_style == "binary":
                        detailed_results["reference_models"].append({
                            "model": model,
                            "output": judgment
                        })
                    else:  # score-based
                        judgment_a, judgment_b = judgment
                        detailed_results["reference_models"].append({
                            "model": model,
                            "output_a": judgment_a,
                            "output_b": judgment_b
                        })
                detailed_results["aggregator"] = {
                    "model": args.aggregator_model,
                    "output": aggregator_judgment
                }
                
                # MoA-specific folder structure
                folder_name = f"moa_{len(args.reference_models)}ref_{args.aggregator_model.split('/')[-1]}"
                output_folder = os.path.join(args.output_dir, folder_name, args.evaluation_style)
            else:
                if isinstance(args.model, list):
                    for model, judgment in zip(args.model, judgement):
                        detailed_results["reference_models"].append({
                            "model": model,
                            "output": judgment
                        })
                    # PoLL-specific folder structure
                    folder_name = f"poll_{len(args.model)}models"
                else:
                    detailed_results["reference_models"].append({
                        "model": args.model,
                        "output": judgement
                    })
                    # Single model folder structure
                    folder_name = f"single_{args.model.split('/')[-1]}"
                
                output_folder = os.path.join(args.output_dir, folder_name)

            os.makedirs(output_folder, exist_ok=True)
            file_name = f"{example_id}_results.json"
            with open(os.path.join(output_folder, file_name), "w") as f:
                json.dump(detailed_results, f, indent=2)

            return detailed_results
        

        def get_judgement(example_id, batch, debug=args.debug):
            mult_turn = True if len(batch["text_chosen"]) > 2 else False
            prompt = batch["text_chosen"][0]["content"]
            answer_a = batch["text_chosen"]
            answer_b = batch["text_rejected"]

            # shuffle a and b randomly for position bias
            is_shuffled = np.random.rand() > 0.5
            if is_shuffled:
                answer_a, answer_b = answer_b, answer_a
                winner_text = "B"
                loser_text = "A"
            else:
                winner_text = "A"
                loser_text = "B"

            if len(batch["text_chosen"]) <= 4:  # set up only for 1 or 2 turns
                answer = {'A': answer_a, 'B': answer_b}
                winner, request, judgement = run_judge_pair(
                    prompt, answer, args.model, multi_turn=mult_turn, 
                    model_modifier=model_modifier, use_moa=args.moa,
                    evaluation_style=args.evaluation_style, debug=args.debug, temperature=args.temperature
                )

                # handle voting for PoLL
                if isinstance(winner, list) and not args.moa:
                    if debug:
                        print(winner)
                    winner = max(set(winner), key=winner.count)
                
                # Save detailed results for both MoA and non-MoA cases
                detailed_results = save_detailed_results(example_id, prompt, answer_a, answer_b, winner, judgement, args, is_shuffled)

                # Process the result
                processed_result = process_result(detailed_results['final_preference'], detailed_results['gold_preference'])
                
                return processed_result, detailed_results
            else:
                return 0.5, {"final_preference": 0.5, "gold_preference": 0.5, "example": {"id": example_id}}
            
 
        with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
            # Map 'my_function' across the vector, executing in parallel using threads
            # results = list(executor.map(get_judgement, dataset))

            # Progress bar version
            results = [None] * len(dataset)  # Preallocate results list
            done_tasks = 0  # Counter for completed tasks
            
            if args.moa:
                folder_name = f"moa_{len(args.reference_models)}ref_{args.aggregator_model.split('/')[-1]}"
                output_folder = os.path.join(args.output_dir, folder_name, args.evaluation_style)
            elif isinstance(args.model, list):
                folder_name = f"poll_{len(args.model)}models"
                output_folder = os.path.join(args.output_dir, folder_name)
            else:
                folder_name = f"single_{args.model.split('/')[-1]}"
                output_folder = os.path.join(args.output_dir, folder_name)

            os.makedirs(output_folder, exist_ok=True)

            # Load existing results
            if args.moa:
                folder_name = f"moa_{len(args.reference_models)}ref_{args.aggregator_model.split('/')[-1]}"
                existing_results = load_existing_results(args, folder_name, args.evaluation_style)
            else:
                folder_name = f"single_{args.model.split('/')[-1]}" if not isinstance(args.model, list) else f"poll_{len(args.model)}models"
                existing_results = load_existing_results(args, folder_name)

            # Create a set of IDs for quick lookup
            existing_ids = set(existing_results.keys())

            # Create lists to store results and processed IDs
            results = []
            processed_ids = []
            all_detailed_results = {}
            
            # Process existing results
            for i, example in enumerate(dataset):
                if example['id'] in existing_ids:
                    result = existing_results[example['id']]
                    results.append(result)
                    processed_ids.append(i)
                    all_detailed_results[example['id']] = result

            # Determine which examples need to be processed
            new_examples = [ex for i, ex in enumerate(dataset) if i not in processed_ids]

            done_tasks = len(existing_results)
            update_progress_bar(done_tasks, len(dataset))

            # Process new examples
            with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
                future_to_index = {executor.submit(get_judgement, ex['id'], ex, args): i for i, ex in enumerate(new_examples)}

                for future in as_completed(future_to_index):
                    index = future_to_index[future]
                    processed_result, detailed_result = future.result()
                    results.append(processed_result)
                    all_detailed_results[detailed_result['example']['id']] = detailed_result
                    done_tasks += 1
                    update_progress_bar(done_tasks, len(dataset))

            # Print newline after progress bar
            print()  
    else:
        ############################
        # Run model weights with vllm
        ############################

        def format_judgements(batch, optional_chat_template=None):
            # TODO expand this to include fastchat chat templates if needed
            mult_turn = True if len(batch["text_chosen"]) > 2 else False
            prompt = batch["text_chosen"][0]["content"]
            answer_a = batch["text_chosen"]
            answer_b = batch["text_rejected"]

            # shuffle a and b randomly for position bias
            is_shuffled = np.random.rand() > 0.5
            if is_shuffled:
                answer_a, answer_b = answer_b, answer_a

            system_prompt, user_prompt = format_judge_answers(
                prompt, answer_a, answer_b, multi_turn=mult_turn, model_modifier=model_modifier
            )

            if optional_chat_template is not None:
                optional_chat_template.set_system_message(system_prompt)
                optional_chat_template.messages = []
                optional_chat_template.append_message(optional_chat_template.roles[0], user_prompt)
                optional_chat_template.append_message(optional_chat_template.roles[1], None)
                prompt = optional_chat_template.get_prompt()
            elif model_modifier:
                messages = [
                    {"role": "user", "content": user_prompt},
                ]
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            batch["text"] = prompt
            batch["is_shuffled"] = is_shuffled
            return batch

        # format the dataset for the model, with optional fastchat templating
        if args.chat_template is not None:
            chat_template = get_conv_template(args.chat_template)
        else:
            chat_template = None
        dataset_prompts = dataset.map(format_judgements, fn_kwargs={"optional_chat_template": chat_template})

        # collect texts of dataset in list
        prompts = dataset_prompts["text"]
        is_shuffled = dataset_prompts["is_shuffled"]

        # generate
        logger.info("*** Run inference ***")
        outputs = model.generate(prompts, sampling_params)
        logger.info("*** Inference done ***")

        answers = [o.outputs[0].text for o in outputs]
        winners = [process_judgement(a, is_prometheus=is_prometheus) for a in answers]

        def process_shuffled(win, shuffle):
            if shuffle:
                winner_text = "B"
                loser_text = "A"
            else:
                winner_text = "A"
                loser_text = "B"

            if win == winner_text:
                return 1
            elif win == loser_text:
                return 0
            else:  # if "error"
                return 0.5  # effectively a tie

        results = [process_shuffled(w, s) for w, s in zip(winners, is_shuffled)]

    ############################
    # Print & process results
    ############################
    # add column for results for easy printing
    out_dataset = dataset.add_column("results", results)

    # add subsets back (removed so it's not handled by cuda)
    #out_dataset = out_dataset.add_column("subset", subsets) 
    #out_dataset = out_dataset.add_column("id", ids)

    # model name concat if list
    if isinstance(args.model, list):
        if args.moa:
            model_name = f"MoA/{args.aggregator_model}"
        else:
            model_name = "_".join(args.model)
            model_name = "PoLL/" + model_name
    else:
        model_name = args.model
    # if model in openai or Anthropic list, append org to model name
    if args.model in OPENAI_MODEL_LIST:
        model_name = "openai/" + model_name
    elif args.model in ANTHROPIC_MODEL_LIST:
        model_name = "anthropic/" + model_name
    elif args.model in GEMINI_MODEL_LIST:
        model_name = "google/" + model_name

    # get core dataset
    results_grouped = {}
    results_grouped["model"] = model_name
    results_grouped["model_type"] = model_type
    results_grouped["chat_template"] = args.chat_template
    if args.moa:
        results_grouped["reference_models"] = args.reference_models
        results_grouped["aggregator_model"] = args.aggregator_model

    # print per subset and log into results_grouped file
    present_subsets = np.unique(subsets)
    for subset in present_subsets:
        subset_dataset = out_dataset.filter(lambda example: example["subset"] == subset)
        num_correct = sum(subset_dataset["results"])
        num_total = len(subset_dataset["results"])
        print(f"{subset}: {num_correct}/{num_total} ({num_correct/num_total})")
        results_grouped[subset] = num_correct / num_total

    logger.info(f"*** Evaluation style: {args.evaluation_style} ***")
    if args.moa:
        logger.info(f"Reference models {args.reference_models}, Aggregator model {args.aggregator_model}")
    else:
        logger.info(f"Model: {model_name}")

    # log leaderboard aggregated results
    if not args.pref_sets:
        if args.debug or args.subset:
            results_leaderboard = calculate_scores_per_section(EXAMPLE_COUNTS, SUBSET_MAPPING, results_grouped, equal_counts=True)
        else:
            results_leaderboard = calculate_scores_per_section(EXAMPLE_COUNTS, SUBSET_MAPPING, results_grouped)
            
        average = sum(results_leaderboard.values()) / len(results_leaderboard)
        print("Average", round(average*100, 1), end="  ")
        # print the key and value, value is rounded to 2 decimal places
        for key, value in results_leaderboard.items():
            print(key, round(value*100, 1), end="  ")
        print()
            
    ############################
    # Upload results to hub
    #############################
    sub_path = "eval-set/" if not args.pref_sets else "pref-sets/"
    results_url = save_to_hub(
        results_grouped,
        model_name,
        sub_path,
        args.debug,
        local_only=args.do_not_save,
        save_metrics_for_beaker=not args.disable_beaker_save,
    )
    if not args.do_not_save:
        logger.info(f"Uploaded reward model results to {results_url}")

    logger.info("Not uploading chosen-rejected text with scores due to model compatibility")

    ############################
    # Save per-prompt results to hub
    ############################
    # create new json with scores and upload
    scores_dict = out_dataset.to_dict()
    scores_dict["model"] = model_name
    scores_dict["model_type"] = model_type

    sub_path_scores = "eval-set-scores/" if not args.pref_sets else "pref-sets-scores/"

    scores_url = save_to_hub(scores_dict, model_name, sub_path_scores, args.debug, local_only=args.do_not_save)
    logger.info(f"Uploading chosen-rejected text with scores to {scores_url}")


if __name__ == "__main__":
    main()
