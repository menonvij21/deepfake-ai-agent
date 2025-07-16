from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from config import BATCH_SIZE, IMG_SIZE, TRAIN_DIR, VAL_DIR, TEST_DIR

def get_loaders():
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])

    train_data = datasets.ImageFolder(TRAIN_DIR, transform=transform)
    val_data = datasets.ImageFolder(VAL_DIR, transform=transform)
    test_data = datasets.ImageFolder(TEST_DIR, transform=transform)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader
