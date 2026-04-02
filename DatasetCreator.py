import os

# Update these paths to your actual locations
partition_file = '/mnt/datadrive/inpaint/CelebA/Eval/list_eval_partition.txt'
image_dir = '/mnt/datadrive/inpaint/CelebA/test'
output_dir = '/home/snuc/Desktop/HINTwithSymmetry/HINT/dataset'
test_mask_dir = '/mnt/datadrive/inpaint/iregularmask/test_mask/mask/testing_mask_dataset'
train_mask_dir = '/mnt/datadrive/inpaint/iregularmask/irregular_mask/disocclusion_img_mask'
# 1. Processing Image Partitions (Train/Val/Test)
train_list, val_list, test_list = [], [], []

print("Reading partition file...")
with open(partition_file, 'r') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) < 2: continue
        
        filename, split_id = parts[0], parts[1]
        full_path = os.path.abspath(os.path.join(image_dir, filename))
        
        if split_id == '0':
            train_list.append(full_path)
        elif split_id == '1':
            val_list.append(full_path)
        elif split_id == '2':
            test_list.append(full_path)

# 2. Scanning Mask Folders
print("Generating mask lists from folders...")
def get_all_mask_paths(mask_dir):
    paths = []
    if os.path.exists(mask_dir):
        for f in os.listdir(mask_dir):
            if f.endswith(('.jpg', '.png', '.jpeg')):
                paths.append(os.path.abspath(os.path.join(mask_dir, f)))
    return sorted(paths)

train_masks = get_all_mask_paths(train_mask_dir)
test_masks  = get_all_mask_paths(test_mask_dir)

# 3. Saving all flist files
def save_flist(flist, name):
    save_path = os.path.join(output_dir, name)
    with open(save_path, 'w') as f:
        f.write('\n'.join(flist))
    print(f"Saved: {save_path} ({len(flist)} entries)")

save_flist(train_list, 'celeba_train.flist')
save_flist(val_list, 'celeba_val.flist')
save_flist(test_list, 'celeba_test.flist')
save_flist(train_masks, 'masks_train.flist')
save_flist(test_masks, 'masks_test.flist')

print("\nAll flist files generated successfully!")
