deepspeed --include localhost:0 ./train_lora.py \
  --model_name_or_path /workspace/Sequence-Scheduling/ckpts/opt-125m \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --data_path ./dummy_conversation.json \
  --bf16 False \
  --output_dir /workspace/Sequence-Scheduling/ckpts \
  --num_train_epochs 1 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 1 \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 1200 \
  --save_total_limit 5 \
  --learning_rate 5e-4 \
  --weight_decay 0. \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "constant" \
  --logging_steps 1 \
  --tf32 False \
  --model_max_length 256 \
  --deepspeed ./deepspeed-config.json \
  --report_to none