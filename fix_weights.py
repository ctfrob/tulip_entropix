import os

weights_dir = 'weights/70B-Instruct/'
files = sorted(os.listdir(weights_dir))
for file in files:
    os.rename(
        os.path.join(weights_dir, file),
        os.path.join(weights_dir, file.zfill(50))
    )