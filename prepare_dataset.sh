pretrained_mask_sythesis_model_path="ss"
vgg_19_path="Models/imagenet-vgg-verydeep-19.mat"
pretrained_mask_removal_model_path="Models/srdplus-pretrained"
images_dir="dataset/images_for_test"
image_split_size="640,480"
output_dataset_folder="generated_dataset"

temp_dir=".temp"
mkdir -p $temp_dir
temp_images_dir="$temp_dir/images"
shadows_dir="$temp_dir/shadows"
splitted_images_path="$temp_dir/splitted_images"
mkdir -p $temp_images_dir $shadows_dir $splitted_images_path
cp $images_dir/* $temp_images_dir
images_dir=$temp_images_dir

# version1 dataset
v1dataset_path="dataset/datasetv1"
v2dataset_path="dataset/datasetv2"

# generate shadow masks for images with _mask suffix
python detect_shadows.py --model $pretrained_mask_removal_model_path --vgg_19_path $vgg_19_path --result_dir $images_dir --input_dir $images_dir

# split images into smaller images
python helpers/splitter_and_merger.py $images_dir $splitted_images_path split $image_split_size

# seperate shadows(train_B), images that has shadows(train_A) and non-shadows(train_C)
python helpers/get_non_maskeds.py $splitted_images_path $v1dataset_path

cp -r $v1dataset_path/train_C $v2dataset_path/train_A
# generate synthetic dataset
python shadow_synthesis.py $v1dataset_path/train_C $v1dataset_path/train_B $v2dataset_path --create-dataset








