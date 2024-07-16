import os
import json
import numpy as np
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from rewardbench.constants import EXAMPLE_COUNTS, SUBSET_MAPPING
from rewardbench.utils import calculate_scores_per_section
from rewardbench import load_eval_dataset
from fastchat.conversation import get_conv_template

def load_json_file(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def process_file(file_path):
    data = load_json_file(file_path)
    # Extract ID from filename
    file_name = os.path.basename(file_path)
    id = int(file_name.split('_')[0])
    return id, data['final_preference'], data['gold_preference']

def evaluate_with_processors(folder_path, num_processors, id_to_subset):
    all_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('_results.json')]
    results = {}

    with ProcessPoolExecutor(max_workers=num_processors) as executor:
        future_to_file = {executor.submit(process_file, file): file for file in all_files}
        for future in as_completed(future_to_file):
            file = future_to_file[future]
            try:
                id, final_pref, gold_pref = future.result()
                subset = id_to_subset.get(id, "Unknown")
                results[id] = {
                    'final_preference': final_pref,
                    'gold_preference': gold_pref,
                    'subset': subset
                }
            except Exception as exc:
                print(f'{file} generated an exception: {exc}')

    return results

def calculate_metrics(results):
    # Group results by subset
    subsets = defaultdict(list)
    for id, data in results.items():
        subsets[data['subset']].append((data['final_preference'], data['gold_preference']))

    # Calculate accuracy for each subset
    results_grouped = {}
    for subset, subset_results in subsets.items():
        correct = sum(1 for fp, gp in subset_results if fp == gp)
        total = len(subset_results)
        results_grouped[subset] = correct / total if total > 0 else 0

    # Use calculate_scores_per_section
    leaderboard_scores = calculate_scores_per_section(EXAMPLE_COUNTS, SUBSET_MAPPING, results_grouped)
    
    # Calculate overall average
    average = sum(leaderboard_scores.values()) / len(leaderboard_scores)
    
    return leaderboard_scores, average

def main():
    folder_path = "/usr/project/xtmp/rx55/projects/moa-eval/results/rewardBench/moa_binary"
    results = defaultdict(list)

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
    id_to_subset = dict(zip(dataset['id'], subsets))

    for num_processors in range(1, 7):
        for _ in range(5):  # Run 5 times for each processor count
            processed_results = evaluate_with_processors(folder_path, num_processors, id_to_subset)
            leaderboard_scores, average = calculate_metrics(processed_results)
            results[num_processors].append((leaderboard_scores, average))

    # Calculate and print average metrics for each processor count
    for num_processors, scores_list in results.items():
        avg_scores = defaultdict(list)
        avg_overall = []
        for leaderboard_scores, average in scores_list:
            for key, value in leaderboard_scores.items():
                avg_scores[key].append(value)
            avg_overall.append(average)

        print(f"Processors: {num_processors}")
        for key in avg_scores:
            mean = np.mean(avg_scores[key])
            std = np.std(avg_scores[key])
            print(f"  {key}: {mean:.4f} ± {std:.4f}")
        overall_mean = np.mean(avg_overall)
        overall_std = np.std(avg_overall)
        print(f"  Overall Average: {overall_mean:.4f} ± {overall_std:.4f}")
        print()

if __name__ == "__main__":
    main()