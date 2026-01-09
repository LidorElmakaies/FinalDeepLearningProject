import zipfile
import shutil
from pathlib import Path
from tqdm import tqdm
import os

# Tree ID splits
VAL_TREES = {"TR007", "TR008", "TR009", "TR040", "TR041", "TR042"}
TEST_TREES = {
    "TR001",
    "TR002",
    "TR018",
    "TR019",
    "TR021",
    "TR087",
    "TR094",
    "TR100",
    "TR103",
    "TR107",
    "TR109",
    "TR115",
    "TR116",
    "TR119",
    "TR156",
    "TR157",
    "TR158",
    "TR159",
    "TR160",
}

# Category mapping
HEALTHY = {"Non_infected"}
SICK = {"Badly_damaged", "Dead", "Infected"}

# Blacklisted corrupted images
BLACKLIST = {
    "TR037-1-B.png",
    "TR029-1-B.png",
    "TR188-1-B.png",
    "TR071-3-B.png",
    "TR142-1-B.png",
    "TR165-1-B.png",
    "TR157-2-B.png",
}


def prepare_thermal_dataset(
    zip_path="data/Date Palm Tree Data Set.zip", output_dir="data"
):
    print(f"Extracting {zip_path}...")
    temp_dir = "temp_extract"
    os.makedirs(temp_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(temp_dir)

    # Find thermal images
    print("Finding thermal images...")
    thermal_images = []
    for root, _, files in os.walk(temp_dir):
        if "THERMAL" in root:
            for file in files:
                if file.endswith("-B.png"):
                    thermal_images.append(os.path.join(root, file))

    print(f"Found {len(thermal_images)} thermal images")

    # Create output structure
    output_path = Path(output_dir)
    for split in ["train", "val", "test"]:
        for label in ["healthy", "sick"]:
            (output_path / split / label).mkdir(parents=True, exist_ok=True)

    # Process images
    stats = {s: {"healthy": 0, "sick": 0} for s in ["train", "val", "test"]}
    skipped_blacklist = 0

    for img_path in tqdm(thermal_images, desc="Processing"):
        filename = os.path.basename(img_path)

        # Skip blacklisted corrupted images
        if filename in BLACKLIST:
            skipped_blacklist += 1
            continue

        tree_id = filename.split("-")[0]
        path_parts = img_path.replace("\\", "/").split("/")

        # Get category
        category = next((p for p in path_parts if p in HEALTHY or p in SICK), None)
        if not category:
            continue

        # Determine split and label
        if tree_id in VAL_TREES:
            split = "val"
        elif tree_id in TEST_TREES:
            split = "test"
        else:
            split = "train"

        label = "healthy" if category in HEALTHY else "sick"

        # Copy file
        dest = output_path / split / label / filename
        shutil.copy2(img_path, dest)
        stats[split][label] += 1

    # Print stats
    print(f"\nOutput: {output_dir}")
    print(
        f"Train - Healthy: {stats['train']['healthy']}, Sick: {stats['train']['sick']}"
    )
    print(f"Val   - Healthy: {stats['val']['healthy']}, Sick: {stats['val']['sick']}")
    print(f"Test  - Healthy: {stats['test']['healthy']}, Sick: {stats['test']['sick']}")
    print(f"Skipped (blacklisted): {skipped_blacklist}")

    # Cleanup
    shutil.rmtree(temp_dir)
    print("Done!")

