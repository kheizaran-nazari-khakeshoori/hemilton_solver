"""Project entry point for running batch QAP experiments.

Run this file to execute run_batch_experiments.main, which iterates over
all generated instances and writes aggregated metrics to batch_results.csv.
"""

from run_batch_experiments import main as run_batch_experiments


if __name__ == "__main__":
	run_batch_experiments()
