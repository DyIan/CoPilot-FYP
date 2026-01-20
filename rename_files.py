import os
from sklearn.model_selection import train_test_split


def rename_files(input_folder, output_folder):

    # Load in images as unchanged
    files = sorted([f for f in os.listdir(input_folder) if f.endswith(".png")])
    print(f"Found {len(files)} in {input_folder}")


    for i, filename in enumerate(files, start=1):
        source_path = os.path.join(input_folder, filename)
        new_filename = f"{i}.png"
        dest_path = os.path.join(output_folder, new_filename)

        os.rename(source_path, dest_path)

#rename_files(r"data\rgb", r"data\rgb_renamed")
def save_split(files, filename):
    with open(filename, "w") as f:
        for name in files:
            f.write(name + "\n")

def split_data():
    rgb_dir = r"data/rgb"
    mask_dir = r"data/mask"

    rgb_files = set(os.listdir(rgb_dir))
    mask_files = set(os.listdir(mask_dir))

    pairs = sorted(list(rgb_files & mask_files))

    print("Total Pairs: ", len(pairs))

    train_files, temp_files = train_test_split(
        pairs,
        test_size=0.3,
        random_state=99
    )

    val_files, test_files = train_test_split(
        temp_files,
        test_size=0.5,
        random_state=99
    )

    print(f"Training: {len(train_files)}\t Val: {len(val_files)}\t Test: {len(test_files)}")

    save_split(train_files, "data/split/train.txt")
    save_split(val_files, "data/split/val.txt")
    save_split(test_files, "data/split/test.txt")

split_data()
