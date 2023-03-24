python src/splitter_and_merger.py dataset/images dataset/splitted_images split "640,480"
#python src/splitter_and_merger.py dataset/shadows dataset/splitted_shadows split "640,480"
python predict.py dataset/splitted_images dataset/splitted_predictions
python src/splitter_and_merger.py dataset/splitted_predictions dataset/merged_predictions merge "640,480"

python train_ss.py \
--task Models/srdplus-pretrained; \
--data_dir dataset/ \
--use_gpu -1 # <0 for CPU \
--is_training 1 # 0 for testing \

python train_sr.py --data_dir dataset --use_gpu -1 --continue_training --task Models/srdplus-pretrained


