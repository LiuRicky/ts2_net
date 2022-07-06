CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7,8
DATA_PATH=[your data path]
python -m torch.distributed.launch --nproc_per_node=8 --master_port 29665 \
main_task_retrieval.py --do_train --eval_in_train --num_thread_reader=12 \
--epochs=5 --batch_size=128 --n_display=50 \
--data_path ${DATA_PATH}/vatex_data \
--features_path ${DATA_PATH}/compressed_videos \
--output_dir ckpts/vatex \
--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 64 \
--datatype vatex \
--feature_framerate 1 --coef_lr 1e-3 \
--freeze_layer_num 0 --slice_framepos 2 \
--loose_type --linear_patch 2d --sim_header seqTransf \
--pretrained_clip_name ViT-B/32