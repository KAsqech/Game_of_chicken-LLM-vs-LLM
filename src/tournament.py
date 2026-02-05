from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")

if API_KEY is None:
    raise RuntimeError("OPENAI_API_KEY not set")

def run_tournament():
    print("Running tournament")