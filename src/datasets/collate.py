from torch.utils.data import DataLoader

def create_loader(ds, batch_size=32, shuffle=True, num_workers=0):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=False)
