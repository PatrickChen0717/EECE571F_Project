from torch.utils.data import DataLoader
from dataloader import KeypointDataset
import glob

yaml_paths = glob.glob(r"C:\Users\Patrick\Documents\eece571F\SurgPose_dataset\**\**\keypoints_left.yaml")
print("num yamls:", len(yaml_paths))

ds = KeypointDataset(
    yaml_paths=yaml_paths,
    kpt_ids=5,
    normalize=True
)

dl = DataLoader(ds, batch_size=32, shuffle=True, num_workers=0)

batch = next(iter(dl))
print(batch["x"].shape)  # (B, seq_len, 5, 2)
print(batch["y"].shape)  # (B, pred_len, 5, 2)