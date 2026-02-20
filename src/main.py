from dotenv import load_dotenv
from tournament import run_experiment, run_both_conditions
from analysis import summarize_results
import os

load_dotenv()

def main():
    print(run_both_conditions())

if __name__ == "__main__":
    main()