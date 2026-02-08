import pandas as pd
import os

# Create directory
os.makedirs("data/harmbench", exist_ok=True)

# 1. Download HarmBench Prompts
print("Fetching HarmBench prompts...")
url = "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/data/behavior_datasets/harmbench_behaviors_text_all.csv"
df = pd.read_csv(url)

# 2. Format for the Agent (VeRL requires these columns)
agent_data = pd.DataFrame()
agent_data['prompt'] = df['Behavior']  # The harmful questions
agent_data['ability'] = "safety"
agent_data['reward_model'] = "none"
agent_data['extra_info'] = "{}"

# 3. Save as Parquet (The only format the agent reads)
agent_data.to_parquet("data/harmbench/test.parquet")
# Create a dummy train file so the script doesn't error out
agent_data.head(5).to_parquet("data/harmbench/train.parquet")

print(f"Ready: 'data/harmbench/test.parquet' created with {len(agent_data)} prompts.")