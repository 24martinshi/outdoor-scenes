from waymo_dataset_val import MyDataset

dataset = MyDataset()
print(len(dataset))

item = dataset[400]
jpg = item['jpg']
txt = item['txt']
hint = item['hint']
print(txt)
print(jpg.shape)
print(hint.shape)
