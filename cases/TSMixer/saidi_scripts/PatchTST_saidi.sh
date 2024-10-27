export CUDA_VISIBLE_DEVICES=0

model_name=PatchTST

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ../../zhongyesaidi \
  --data_path lab_hourly_average.csv \
  --model_id PatchTST_saidi_24_1 \
  --model $model_name \
  --data saidi \
  --features M \
  --seq_len 24 \
  --label_len 1 \
  --pred_len 1 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 6 \
  --dec_in 6 \
  --c_out 6 \
  --des 'Exp' \
  --itr 1