import os

# Update these paths to your actual locations
partition_file = './datasets/celeba/list_eval_partition.txt' 
image_dir = './datasets/celeba/img_align_celeba' 
output_dir = './datasets/celeba'

# Create lists for each split
train_list, val_list, test_list = [], [], []

print("Readings partition file...")
with open(partition_file, 'r') as f:
    for line in f:
        # Expected format: "filename partition_id" (e.g. "000001.jpg 0")
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

print(f"Stats: Train={len(train_list)}, Val={len(val_list)}, Test={len(test_list)}")

# Save flist files
def save_flist(flist, name):
    with open(os.path.join(output_dir, name), 'w') as f:
        f.write('\n'.join(flist))

save_flist(train_list, 'celeba_train.flist')
save_flist(val_list, 'celeba_val.flist')
save_flist(test_list, 'celeba_test.flist')

print(f"Success! Generated flists in {output_dir}")
