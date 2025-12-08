filter_difficulty="Easy"
model_name="Qwen/Qwen3-4B-Instruct-2507"
YOUR_PORT_NUMBER=8000
max_train_samples=600
batch_size=64
group_size=8

# docker run -it -p $YOUR_PORT_NUMBER:8080 volcengine/sandbox-fusion:server-20250609
# docker run -it -p 8001:8080 volcengine/sandbox-fusion:server-20250609

python train.py \
    filter_difficulty=$filter_difficulty \
    sandbox_url="http://localhost:$YOUR_PORT_NUMBER/run_code" \
    log_path="outputs/rl-leetcode/$model_name/$filter_difficulty" \
    epochs=5 \
    max_train_samples=$max_train_samples \
    batch_size=$batch_size \
    group_size=$group_size \
    model_name=$model_name