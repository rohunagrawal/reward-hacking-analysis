filter_by="init_rollout_entropy"
entropy_bin=1
dataset_path="data/leetcode_entropy"
model_name="meta-llama/Llama-3.2-3B"
YOUR_PORT_NUMBER=8000
max_train_samples=500
batch_size=64
group_size=8
reward_type="g"   # g, f_g, 100f_g

# docker run -it -p $YOUR_PORT_NUMBER:8080 volcengine/sandbox-fusion:server-20250609

python train.py \
    filter_by=$filter_by \
    filter_entropy_bin=$entropy_bin \
    dataset_path=$dataset_path \
    sandbox_url="http://localhost:$YOUR_PORT_NUMBER/run_code" \
    log_path="outputs/rl-leetcode-entropy/$model_name/bin_$entropy_bin/$reward_type" \
    epochs=5 \
    max_train_samples=$max_train_samples \
    batch_size=$batch_size \
    group_size=$group_size \
    model_name=$model_name \
    reward_type=$reward_type
