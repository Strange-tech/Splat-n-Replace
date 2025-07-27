SCENE_NAME=BlendedMVS/green_houses
DATASET_PATH=/root/autodl-tmp/data/$SCENE_NAME
PREFIX=/Splat-n-Replace

# # COLMAP
# echo "Running COLMAP feature extraction and matching..."

# colmap feature_extractor --database_path $DATASET_PATH/database.db --image_path $DATASET_PATH/images
# colmap exhaustive_matcher --database_path $DATASET_PATH/database.db
# mkdir $DATASET_PATH/sparse
# colmap mapper --database_path $DATASET_PATH/database.db --image_path $DATASET_PATH/images --output_path $DATASET_PATH/sparse
 
# # --text "sofa. pillow. armchair. ottoman. table. vase. flower. ground. lamp. book. window. wall. painting. blanket. bag. ceiling." \
# # --text "ground. house. car. tree. grass."
# # --text "sofa. table. floor. wall. ceiling. screen. chair. ottoman. door. pillow."
# # --text "table. floor. wall. window. chair. door. picture. shelf. vase."
# # --text "screen. world map. plant. pot. whiteboard. carpet. creature. wall. chair. table. door. sofa."

# cd $PREFIX/third_party/Grounded-SAM-2
# echo "Running Grounded-SAM-2 tracking..."
# python grounded_sam2_tracking_demo_with_continuous_id_gd1.5.py \
#     --text "house." \
#     --video_dir $DATASET_PATH/images/ \
#     --output_dir /root/autodl-tmp/output/$SCENE_NAME/

# cd $PREFIX/utils
# echo "Converting Grounded-SAM masks..."
# python mask_npy2pt.py \
#     --input_dir /root/autodl-tmp/output/$SCENE_NAME/mask_data/ \
#     --output_dir $DATASET_PATH/sam_masks/

# cd $PREFIX
# echo "Training Vanilla 3DGS..."
# python train_scene.py \
#     --source_path $DATASET_PATH \
#     --model_path /root/autodl-tmp/3dgs_output/$SCENE_NAME \
#     --iterations 30000

cd $PREFIX
echo "Training Contrastive GS Feature..."
python get_scale.py \
    --image_root $DATASET_PATH \
    --model_path /root/autodl-tmp/3dgs_output/$SCENE_NAME

python train_contrastive_feature.py \
    --model_path /root/autodl-tmp/3dgs_output/$SCENE_NAME \
    --iterations 10000 \
    --num_sampled_rays 1000

