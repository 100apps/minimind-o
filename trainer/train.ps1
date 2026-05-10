# ==================== Full dataset training pipeline ====================
# Suggested full training pipeline (Dense, 4x GPU)
# Note: torchrun is not supported on Windows (libuv issue).
#       Use direct python execution for single-GPU training.

# $env:CUDA_VISIBLE_DEVICES="0,1,2,3"
# uv run python train_sft_omni.py --learning_rate 5e-4 --data_path ..\dataset\sft_t2a.parquet --epochs 6 --batch_size 32 --use_compile 0 --from_weight llm --save_weight sft_omni --use_wandb --use_moe 0 --num_workers 0
# uv run python train_sft_omni.py --learning_rate 5e-4 --data_path ..\dataset\sft_a2a.parquet --epochs 1 --batch_size 32 --use_compile 0 --from_weight sft_omni --save_weight sft_omni --max_seq_len 1024 --mode audio_proj --use_wandb --use_moe 0 --num_workers 0
# uv run python train_sft_omni.py --learning_rate 5e-5 --data_path ..\dataset\sft_a2a.parquet --epochs 3 --batch_size 32 --use_compile 0 --from_weight sft_omni --save_weight sft_omni --max_seq_len 1024 --use_wandb --use_moe 0 --num_workers 0
# uv run python train_sft_omni.py --learning_rate 5e-5 --data_path ..\dataset\sft_i2t.parquet --epochs 1 --batch_size 32 --use_compile 0 --from_weight sft_omni --save_weight sft_omni --max_seq_len 768 --mode vision_proj --use_wandb --use_moe 0 --num_workers 0
# uv run python train_sft_omni.py --learning_rate 5e-6 --data_path ..\dataset\sft_i2t.parquet --epochs 1 --batch_size 32 --use_compile 0 --from_weight sft_omni --save_weight sft_omni --max_seq_len 768 --use_wandb --use_moe 0 --num_workers 0
# uv run python train_sft_omni.py --learning_rate 5e-6 --data_path ..\dataset\sft_a2a.parquet --epochs 1 --batch_size 32 --use_compile 0 --from_weight sft_omni --save_weight sft_omni --max_seq_len 1024 --use_wandb --use_moe 0 --num_workers 0
# uv run python train_sft_omni.py --learning_rate 5e-6 --data_path ..\dataset\sft_i2t.parquet --epochs 1 --batch_size 32 --use_compile 0 --from_weight sft_omni --save_weight sft_omni --max_seq_len 768 --mode vision_proj --use_wandb --use_moe 0 --num_workers 0


# Suggested full training pipeline (MoE, 4x GPU)
# $env:CUDA_VISIBLE_DEVICES="0,1,2,3"
# uv run python train_sft_omni.py --learning_rate 5e-4 --data_path ..\dataset\sft_t2a.parquet --epochs 6 --batch_size 32 --use_compile 0 --from_weight llm --save_weight sft_omni --use_wandb --use_moe 1 --num_workers 0
# uv run python train_sft_omni.py --learning_rate 5e-4 --data_path ..\dataset\sft_a2a.parquet --epochs 1 --batch_size 32 --use_compile 0 --from_weight sft_omni --save_weight sft_omni --max_seq_len 1024 --mode audio_proj --use_wandb --use_moe 1 --num_workers 0
# uv run python train_sft_omni.py --learning_rate 5e-5 --data_path ..\dataset\sft_a2a.parquet --epochs 3 --batch_size 32 --use_compile 0 --from_weight sft_omni --save_weight sft_omni --max_seq_len 1024 --use_wandb --use_moe 1 --num_workers 0
# uv run python train_sft_omni.py --learning_rate 5e-5 --data_path ..\dataset\sft_i2t.parquet --epochs 1 --batch_size 32 --use_compile 0 --from_weight sft_omni --save_weight sft_omni --max_seq_len 768 --mode vision_proj --use_wandb --use_moe 1 --num_workers 0
# uv run python train_sft_omni.py --learning_rate 5e-6 --data_path ..\dataset\sft_i2t.parquet --epochs 1 --batch_size 32 --use_compile 0 --from_weight sft_omni --save_weight sft_omni --max_seq_len 768 --use_wandb --use_moe 1 --num_workers 0
# uv run python train_sft_omni.py --learning_rate 5e-6 --data_path ..\dataset\sft_a2a.parquet --epochs 1 --batch_size 32 --use_compile 0 --from_weight sft_omni --save_weight sft_omni --max_seq_len 1024 --use_wandb --use_moe 1 --num_workers 0
# uv run python train_sft_omni.py --learning_rate 5e-6 --data_path ..\dataset\sft_i2t.parquet --epochs 1 --batch_size 32 --use_compile 0 --from_weight sft_omni --save_weight sft_omni --max_seq_len 768 --mode vision_proj --use_wandb --use_moe 1 --num_workers 0


# ==================== (Recommend) Mini dataset training pipeline ====================
# Time estimates below are based on 1x RTX 3090, for reference only and may not be entirely accurate.
$env:CUDA_VISIBLE_DEVICES="0"
uv run python train_sft_omni.py --learning_rate 5e-4 --data_path ..\dataset\sft_t2a_mini.parquet --epochs 1 --batch_size 40 --use_compile 0 --from_weight llm --save_weight sft_zero --max_seq_len 512 --use_wandb --use_moe 0 --num_workers 0
uv run python train_sft_omni.py --learning_rate 5e-4 --data_path ..\dataset\sft_a2a_mini.parquet --epochs 1 --batch_size 40 --use_compile 0 --from_weight sft_zero --save_weight sft_zero --max_seq_len 640 --mode audio_proj --use_wandb --use_moe 0 --num_workers 0
uv run python train_sft_omni.py --learning_rate 2e-5 --data_path ..\dataset\sft_a2a_mini.parquet --epochs 1 --batch_size 16 --use_compile 0 --from_weight sft_zero --save_weight sft_zero --max_seq_len 768 --use_wandb --use_moe 0 --num_workers 0
