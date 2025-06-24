import os
import shutil
import random

# Configuration
source_dir = "datasets/mango-fruits"
output_dir = "datasets/split-mango-fruits"
split_ratios = {'train': 0.7, 'val': 0.15, 'test': 0.15}
random_seed = 42

# Create split directories
for split in split_ratios:
    for class_name in os.listdir(source_dir):
        class_dir = os.path.join(output_dir, split, class_name)
        os.makedirs(class_dir, exist_ok=True)

# Split the files
random.seed(random_seed)

for class_name in os.listdir(source_dir):
    class_path = os.path.join(source_dir, class_name)
    if not os.path.isdir(class_path):
        continue
    
    images = os.listdir(class_path)
    random.shuffle(images)

    total = len(images)
    train_end = int(split_ratios['train'] * total)
    val_end = train_end + int(split_ratios['val'] * total)

    split_files = {
        'train': images[:train_end],
        'val': images[train_end:val_end],
        'test': images[val_end:]
    }

    for split in split_files:
        for img in split_files[split]:
            src_path = os.path.join(class_path, img)
            dst_path = os.path.join(output_dir, split, class_name, img)
            shutil.copy(src_path, dst_path)

print("✅ Dataset split complete.")
