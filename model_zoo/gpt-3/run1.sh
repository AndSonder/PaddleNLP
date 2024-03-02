echo "=========== $FUNCNAME run begin ==========="A
export PYTHONPATH=../../:$PYTHONPATH

log_dir=mylog
rm -rf $log_dir
python -m paddle.distributed.launch --log_dir=./mylog --devices=0,1,2,3 tools/auto.py \
    -c ./ppfleetx/configs/nlp/gpt/auto/pretrain_gpt_345M_single_card.yaml \
    -o Model.hidden_dropout_prob=0 \
    -o Model.attention_probs_dropout_prob=0 \
    -o Model.use_recompute=True \
    -o Global.global_batch_size=4 \
    -o Global.local_batch_size=2 \
    -o Global.micro_batch_size=1 \
    -o Distributed.dp_degree=2 \
    -o Distributed.mp_degree=1 \
    -o Distributed.pp_degree=2 \
    -o Distributed.sharding.sharding_degree=1 \
    -o Distributed.sharding.sharding_stage=1 \
    -o Distributed.pipeline.schedule_mode=1F1B \
    -o Engine.mix_precision.enable=True \
    -o Engine.mix_precision.level="o2" \
    -o Engine.max_steps=30 \
    -o Engine.eval_freq=100000 \
    -o Engine.logging_freq=10 \
        -o Profiler_auto.memory_stats=True \
