python hub/examples/image_retraining/retrain.py \
    --image_dir augumented_data/ \
    --output_graph breed_class_1_224_model.pb \
    --output_labels breed_class_1_224_labels.txt \
    --how_many_training_steps=20000 \
    --tfhub_module https://tfhub.dev/google/imagenet/mobilenet_v1_100_224/quantops/feature_vector/1

    # --learning_rate=0.0001 \
    # --eval_step_interval=100 \
    # --train_batch_size=150 \
    # --validation_batch_size=500 \
    # --validation_percentage=15 \
    # --testing_percentage=15 \
