python tensorflow/tensorflow/examples/label_image/label_image.py \
--graph=breed_recognition_model_graph.pb \
--labels=breed_recognition_model_labels.txt \
--image=pekin.jpg \
--input_layer=Placeholder \
--output_layer=final_result \
--input_width=128 \
--input_height=128 \
--input_mean=0 \
--input_std=255
