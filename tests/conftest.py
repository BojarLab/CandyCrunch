# conftest.py
from tabulate import tabulate
import numpy as np
from collections import defaultdict
import pytest
import json
import os
from datetime import datetime

class ResultCollector:
    def __init__(self):
        self.results = defaultdict(list)
        self.param_names = None
        self.current_dict = None
        self.dict_results = defaultdict(lambda: defaultdict(list))  
        self.log_file = "test_results_log.json"
        self.previous_results = self.load_previous_results()

    def load_previous_results(self):
        if os.path.exists(self.log_file):
            with open(self.log_file, 'r') as f:
                return json.load(f)
        return {}

    def save_current_results(self):
        # Calculate final averages for each test_dict
        final_results = {}
        for dict_name, param_results in self.dict_results.items():
            averaged_results = {
                str(params): np.mean(scores)  # Convert tuple to str for JSON serialization
                for params, scores in param_results.items()
            }
            final_results[dict_name] = {
                'scores': averaged_results,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        
        # Save to log file
        with open(self.log_file, 'w') as f:
            json.dump(final_results, f, indent=4)
    
    def check_performance(self, test_dict_name, param_key, current_score):
        """Check if current score is at least as good as previous best"""
        if test_dict_name in self.previous_results:
            prev_scores = self.previous_results[test_dict_name]['scores']
            param_key_str = str(param_key)
            if param_key_str in prev_scores:
                prev_score = prev_scores[param_key_str]
                if current_score < prev_score:
                    raise AssertionError(
                        f"\nPerformance regression detected for {test_dict_name}!"
                        f"\nPrevious score: {prev_score:.3f}"
                        f"\nCurrent score: {current_score:.3f}"
                        f"\nParameters: {param_key}")

    def add_result(self, params, score):
        test_dict_name = params['test_dict']['name']
        param_key = tuple(
            params[key] if key != 'test_dict' else params['test_dict']['name']
            for key in self.param_names.keys()
        )
        
        # Append the individual score to the list of scores for this parameter combination
        self.dict_results[test_dict_name][param_key[1:]].append(score)
        
        # If we've switched to a new test_dict, print the results for the previous one
        if self.current_dict != test_dict_name and self.current_dict is not None:
            self.print_dict_results(self.current_dict)
        
        self.current_dict = test_dict_name
    
    def print_dict_results(self, dict_name):
        print(f"\n=== Results for {dict_name} ===")
        
        # Get results for this test_dict
        dict_specific_results = self.dict_results[dict_name]
        
        # Generate headers (exclude test_dict since it's the same for all rows)
        headers = list(self.param_names.values())[1:] + ['F1 Score']
        table_data = []
        
        # Calculate averages and sort by F1 score
        averaged_results = {
            params: np.mean(scores) 
            for params, scores in dict_specific_results.items()
        }
        
        # Sort by average F1 score and create table rows
        for params, avg_f1 in sorted(averaged_results.items(), key=lambda x: x[1], reverse=True):
            row = list(params) + [f"{avg_f1:.3f}"]
            table_data.append(row)
        
        print(tabulate(table_data, headers=headers, tablefmt='grid'))
        print()
    
    def get_table(self):
        if not self.dict_results:
            return "No results collected!"
        
        # Print final results for the last test_dict
        if self.current_dict:
            self.print_dict_results(self.current_dict)
        
        self.save_current_results()

        # Print overall results
        print("\n=== Overall Performance Results ===")
        headers = list(self.param_names.values()) + ['F1 Score']
        table_data = []
        
        # Combine results from all test_dicts
        overall_results = defaultdict(list)
        for dict_name, param_results in self.dict_results.items():
            for params, scores in param_results.items():
                overall_key = (dict_name,) + params
                overall_results[overall_key].extend(scores)
        
        # Calculate averages and sort
        averaged_overall = {
            params: np.mean(scores)
            for params, scores in overall_results.items()
        }
        
        for params, avg_f1 in sorted(averaged_overall.items(), key=lambda x: x[1], reverse=True):
            row = list(params) + [f"{avg_f1:.3f}"]
            table_data.append(row)
        
        return tabulate(table_data, headers=headers, tablefmt='grid')

# Rest remains the same
collector = ResultCollector()

@pytest.fixture(scope="session")
def result_collector():
    return collector

def pytest_configure(config):
    config.collector = collector


def pytest_sessionfinish():
    print("\n=== Final Summary ===")
    
    # Print tables for each test_dict
    for dict_name in collector.dict_results.keys():
        collector.print_dict_results(dict_name)
    
    # Print overall results
    print("\n=== Overall Performance Results ===")
    print(collector.get_table())
    print()

    if collector.previous_results:
        print("\n=== Comparison with Previous Run ===")
        for dict_name in collector.dict_results.keys():
            if dict_name in collector.previous_results:
                prev_timestamp = collector.previous_results[dict_name]['timestamp']
                prev_score = collector.previous_results[dict_name]['scores']
                print(f"\n{dict_name} (Previous run: {prev_timestamp}, Score: {prev_score})")