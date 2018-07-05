python tensorflow/tensorflow/examples/label_image/label_image.py \
--graph=breed_class_1_224_model.pb \
--labels=breed_class_1_224_labels.txt \
--image=terier.jpg \
--input_layer=Placeholder \
--output_layer=final_result \
--input_mean=0 \
--input_std=255 \
--input_width=224 \
--input_height=224
