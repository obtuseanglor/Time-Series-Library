export CUDA_VISIBLE_DEVICES=0

model_name=TimesNet

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./SCADA_Dataset/V1/DATA/ \
  --data_path ALL.csv \
  --model_id ETTh1_96_96 \
  --model $model_name \
  --data OURS \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --d_model 16 \
  --d_ff 32 \
  --des 'Exp' \
  --itr 1 \
  --top_k 5 


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./SCADA_Dataset/V1/DATA/ \
  --data_path ALL.csv \
  --model_id ETTh1_96_192 \
  --model $model_name \
  --data OURS \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --d_model 16 \
  --d_ff 32 \
  --des 'Exp' \
  --itr 1 \
  --top_k 5


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./SCADA_Dataset/V1/DATA/ \
  --data_path ALL.csv \
  --model_id ETTh1_96_336 \
  --model $model_name \
  --data OURS \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --d_model 16 \
  --d_ff 32 \
  --des 'Exp' \
  --itr 1 \
  --top_k 5


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./SCADA_Dataset/V1/DATA/ \
  --data_path ALL.csv \
  --model_id ETTh1_96_720 \
  --model $model_name \
  --data OURS \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --d_model 16 \
  --d_ff 32 \
  --des 'Exp' \
  --itr 1 \
  --top_k 5
