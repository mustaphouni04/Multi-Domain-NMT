TRAIN_SET=Splits/neulab_train_set.csv
VALID_SET=Splits/neulab_valid_set.csv
TEST_SET=Splits/neulab_test_set.csv
OUTPUT_DIR=neulab_translation/
TRAIN_BATCH_SIZE=12
TEST_BATCH_SIZE=12
MAX_SOURCE_LENGTH=512 # the max for marian 


python run_translation.py \
--model_name_or_path Helsinki-NLP/opus-mt-en-es \
--do_train \
--do_eval \
--source_lang en \
--target_lang es \
--train_file $TRAIN_SET \
--validation_file $VALID_SET \
--output_dir $OUTPUT_DIR \
--per_device_train_batch_size=$TRAIN_BATCH_SIZE \
--per_device_eval_batch_size=$TEST_BATCH_SIZE \
--overwrite_output_dir \
--test_file $TEST_SET \
--predict_with_generate \
--max_source_length=$MAX_SOURCE_LENGTH
