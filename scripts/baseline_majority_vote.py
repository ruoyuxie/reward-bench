import os
import json
import re
import argparse
import numpy as np
from collections import defaultdict
from rewardbench.constants import EXAMPLE_COUNTS, SUBSET_MAPPING
from rewardbench.utils import calculate_scores_per_section
from rewardbench import load_eval_dataset
from fastchat.conversation import get_conv_template

def load_json_file(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def extract_vote(output):
    match = re.search(r'\[\[(A|B)\]\]', output)
    if match:
        return 1 if match.group(1) == 'A' else 0
    return None

def process_file(file_path):
    data = load_json_file(file_path)
    votes = []
    for model in data['reference_models']:
        vote = extract_vote(model['output'])
        if vote is not None:
            votes.append(vote)
    
    if not votes:
        majority_vote = None
    else:
        majority_vote = 1 if sum(votes) > len(votes) / 2 else 0
    
    return majority_vote, data['final_preference'], data['gold_preference']

def evaluate_dataset(folder_path):
    all_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('_results.json')]
    results = {}

    for file in all_files:
        try:
            id = int(os.path.basename(file).split('_')[0])
            majority_vote, final_preference, gold_preference = process_file(file)
            results[id] = {
                'majority_vote': majority_vote,
                'final_preference': final_preference,
                'gold_preference': gold_preference
            }
        except Exception as exc:
            print(f'{file} generated an exception: {exc}')

    return results

def calculate_metrics(results, id_to_subset, use_majority_vote=True):
    results_grouped = defaultdict(lambda: {'correct': 0, 'total': 0})
    present_subsets = np.unique(list(id_to_subset.values()))
    
    for id, result in results.items():
        subset = id_to_subset.get(id)
        if subset:
            results_grouped[subset]['total'] += 1
            if use_majority_vote:
                if result['majority_vote'] is not None:
                    if result['majority_vote'] == result['gold_preference']:
                        results_grouped[subset]['correct'] += 1
                    elif abs(result['majority_vote'] - result['gold_preference']) == 0.5:
                        results_grouped[subset]['correct'] += 0.5
            else:
                if result['final_preference'] == result['gold_preference']:
                    results_grouped[subset]['correct'] += 1
                elif abs(result['final_preference'] - result['gold_preference']) == 0.5:
                    results_grouped[subset]['correct'] += 0.5

    for subset in present_subsets:
        correct = results_grouped[subset]['correct']
        total = results_grouped[subset]['total']
        accuracy = correct / total if total > 0 else 0
        print(f"{subset}: {correct:.1f}/{total} ({accuracy:.4f})")
        results_grouped[subset] = accuracy

    results_leaderboard = calculate_scores_per_section(EXAMPLE_COUNTS, SUBSET_MAPPING, results_grouped, equal_counts=True)
    average = sum(results_leaderboard.values()) / len(results_leaderboard)
    
    return results_grouped, results_leaderboard, average

def print_results(results_leaderboard, average):
    print("Average", round(average*100, 1), end="  ")
    for key, value in results_leaderboard.items():
        print(key, round(value*100, 1), end="  ")
    print()

def main():
    parser = argparse.ArgumentParser(description="Evaluate baseline majority vote and aggregator performance.")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    args = parser.parse_args()

    folder_path = "/usr/project/xtmp/rx55/projects/moa-eval/results/rewardBench/single_Llama-3-70b-chat-hf"
    
    # Load the dataset to get id to subset mapping
    dataset, subsets = load_eval_dataset(
        core_set=True,
        conv=get_conv_template("raw"),
        custom_dialogue_formatting=True,
        tokenizer=None,
        logger=None,
        keep_columns=["id", "subset"],
        max_turns=4,
    )
    id_to_subset = {id: subset for id, subset in zip(dataset['id'], subsets)}

    results = evaluate_dataset(folder_path)

    print("\nAggregator (final_preference) Results:")
    results_grouped, results_leaderboard, average = calculate_metrics(results, id_to_subset, use_majority_vote=False)
    print_results(results_leaderboard, average)

    print("\nMajority Voting Baseline Results:")
    results_grouped, results_leaderboard, average = calculate_metrics(results, id_to_subset, use_majority_vote=True)
    print_results(results_leaderboard, average)
    
if __name__ == "__main__":
    main()