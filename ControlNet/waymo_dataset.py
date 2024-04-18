import json
import cv2
import numpy as np

from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self):
        self.data = []
        # with open(f'./training/fill50k/prompt.json', 'rt') as f:
        # with open(f'./training/waymo/prompt.json', 'rt') as f:
        with open(f'./training/waymo_t/prompt.json', 'rt') as f:
        # with open(f'./training/waymo_seg/prompt.json', 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        # source = cv2.imread(f'./training/waymo_seg/' + source_filename)
        # target = cv2.imread(f'./training/waymo_seg/' + target_filename)
        
        source = cv2.imread(f'./training/waymo_unif/' + source_filename)
        target = cv2.imread(f'./training/waymo_unif/' + target_filename)
        # source = cv2.imread(f'./training/waymo_t/' + source_filename)
        # target = cv2.imread(f'./training/waymo_t/' + target_filename)
        # source = cv2.imread(f'./training/waymo/' + source_filename)
        # target = cv2.imread(f'./training/waymo/' + target_filename)
        # source = cv2.imread(f'./training/fill50k/' + source_filename)
        # target = cv2.imread(f'./training/fill50k/' + target_filename)

        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        source = source.astype(np.float32) / 255.0

        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)

