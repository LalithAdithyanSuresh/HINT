import os
import random
import glob
import shutil
import numpy as np
from PIL import Image
import subprocess

def run():
    dataset_dir = r"d:\_Projects\HINT\img_align_celeba\img_align_celeba"
    output_inputs = r"d:\_Projects\HINT\HINT\test_inputs"
    output_masks = r"d:\_Projects\HINT\HINT\test_masks"
    results_dir = r"d:\_Projects\HINT\HINT\test_outputs"

    os.makedirs(output_inputs, exist_ok=True)
    os.makedirs(output_masks, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Clean existing temp files if any
    for d in [output_inputs, output_masks, results_dir]:
        for f in glob.glob(os.path.join(d, "*")):
            if os.path.isfile(f):
                os.remove(f)

    all_images = glob.glob(os.path.join(dataset_dir, "*.jpg"))
    if not all_images:
        print(f"Error: No images found in {dataset_dir}.")
        return

    random.shuffle(all_images)
    sampled_images = all_images[:50]

    print(f"Sampled {len(sampled_images)} images. Generating masks...")

    for i, img_path in enumerate(sampled_images):
        # copy image
        base_name = os.path.basename(img_path)
        dest_img_path = os.path.join(output_inputs, base_name)
        shutil.copyfile(img_path, dest_img_path)

        # open image to get size
        img = Image.open(img_path)
        w, h = img.size
        
        # generate random mask
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # random mask size: between 1/4 and 1/2 of dimensions
        mask_w = random.randint(w // 4, w // 2)
        mask_h = random.randint(h // 4, h // 2)
        
        # random position
        mask_x = random.randint(0, max(1, w - mask_w))
        mask_y = random.randint(0, max(1, h - mask_h))
        
        # 255 for masked region
        mask[mask_y:mask_y+mask_h, mask_x:mask_x+mask_w] = 255
        
        # save mask
        mask_img = Image.fromarray(mask)
        # Using same basename to match naming if model matches by name
        dest_mask_path = os.path.join(output_masks, base_name)
        mask_img.save(dest_mask_path)
    
    print("Executing HINT test.py...")
    # cd to the HINT project folder where test.py is located
    current_dir = os.getcwd()
    hint_dir = r"d:\_Projects\HINT\HINT"
    os.chdir(hint_dir)
    os.environ["WANDB_MODE"] = "disabled"
    
    # We call python test.py --input test_inputs --mask test_masks --output test_outputs
    cmd = ["python", "test.py", "--path", "./checkpoints", "--input", "test_inputs", "--mask", "test_masks", "--output", "test_outputs"]
    
    try:
        subprocess.run(cmd, check=True)
        print("Inference completed successfully!")
    except subprocess.CalledProcessError as e:
        print("Error running inference:", e)
    finally:
        os.chdir(current_dir)

if __name__ == "__main__":
    run()
