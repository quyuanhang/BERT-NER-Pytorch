CURRENT_DIR=`pwd`
export BERT_BASE_DIR=$CURRENT_DIR/prev_trained_model/bert-base-chinese
export DATA_DIR=$CURRENT_DIR/datasets
export OUTPUR_DIR=$CURRENT_DIR/outputs
TASK_NAME="aipf2"
#
python run_ner_crf_lstm.py \
  --no_cuda \
  --do_adv \
  --model_type=bert_lstm \
  --model_name_or_path=$BERT_BASE_DIR \
  --task_name=$TASK_NAME \
  --overwrite_cache \
  --do_lower_case \
  --data_dir=$DATA_DIR/${TASK_NAME}/ \
  --train_max_seq_length=128 \
  --eval_max_seq_length=512 \
  --per_gpu_train_batch_size=24 \
  --per_gpu_eval_batch_size=24 \
  --learning_rate=3e-5 \
  --crf_learning_rate=1e-3 \
  --do_train \
  --num_train_epochs=4.0 \
  --logging_steps=-1 \
  --save_steps=-1 \
  --output_dir=$OUTPUR_DIR/${TASK_NAME}_output/ \
  --overwrite_output_dir \
  --seed=42 \
  --do_train \
  --do_predict \
  --predict_checkpoints=160 \
  --predict_input_json=$DATA_DIR/${TASK_NAME}/test.json \
  --predict_output_json=$DATA_DIR/${TASK_NAME}/test2.json
