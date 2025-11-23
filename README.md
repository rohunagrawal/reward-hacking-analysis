# Setup

Set up your favorite virtual environment and then:
```
git clone git@github.com:rohunagrawal/reward-hacking-entropy.git
cd reward-hacking-entropy
pip install -r requirements.txt
```

# Set your tinker API key
```
export TINKER_API_KEY=...
```

# Prepare dataset
```
python data/prep_dataset.py
```

# Reward Function for Coding
- Host [SandBox Fusion](https://bytedance.github.io/SandboxFusion/docs/docs/get-started#local-deployment) (create a sandbox to test code safely):
  - ```docker run -it -p 8000:8000 volcengine/sandbox-fusion:server-20250609``` 
  - Pass in the url when creating the LeetCode() object below.
- Create LeetCode() object: class defined in [leetcode.py](reward_function/leetcode.py)
  - Entry function: ```process_code_result()```. Returns a dictionary with ```is_compilable_reward``` and ```correctness_reward``` 
  - ```is_compilable(completion: str)```: only checking whether the code itself is compilable, not whether the whole code with imports and test cases is compilable. (may happen in sandbox fusion)
  - ```check_correctness()```: use SandBox Fusion. Fall back to prime_code if SandBox Fusion is not available.

# Run an RL training run
```
python train.py
```
