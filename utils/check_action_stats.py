import tensorflow_datasets as tfds
import numpy as np

DATASET_DIR = "/home/aivlab/tensorflow_datasets/so101_pickup/1.0.0"

builder = tfds.builder_from_directory(DATASET_DIR)
dataset = builder.as_dataset(split="train")

dpos_mag, drot_mag, dgripper = [], [], []
for ep in dataset:
    for step in ep["steps"]:
        a = step["action"].numpy()
        dpos_mag.append(np.linalg.norm(a[:3]))
        drot_mag.append(np.linalg.norm(a[3:6]))
        dgripper.append(a[6])

dpos_mag = np.array(dpos_mag)
drot_mag = np.array(drot_mag)
dgripper = np.array(dgripper)

print(f"d_pos  (m)   : max={dpos_mag.max():.4f}, 99pct={np.percentile(dpos_mag,99):.4f}, median={np.median(dpos_mag):.4f}")
print(f"d_rot  (rad) : max={drot_mag.max():.4f}, 99pct={np.percentile(drot_mag,99):.4f}, median={np.median(drot_mag):.4f}")
print(f"d_grip (deg) : max={dgripper.max():.2f}, min={dgripper.min():.2f}, 99pct_abs={np.percentile(np.abs(dgripper),99):.2f}")
print(f"\nClip[-0.03, 0.03] affects {(dpos_mag > 0.03).mean()*100:.1f}% of steps")
