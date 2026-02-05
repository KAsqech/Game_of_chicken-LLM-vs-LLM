from dotenv import load_dotenv
from tournament import run_tournament
from analysis import summarize_results
import os

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")

if API_KEY is None:
    raise RuntimeError("OPENAI_API_KEY not set")

def main():
    print(run_tournament())
    summarize_results()

if __name__ == "__main__":
    main()