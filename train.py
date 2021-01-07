import torch
from path import Path
from torch.utils.data import DataLoader
from torchvision import transforms

from .dataset import *
from .model import *
from .loss import pointnetloss



path = Path("ModelNet10")
folders = [dir for dir in sorted(os.listdir(path)) if os.path.isdir(path/dir)]
classes = {folder: i for i, folder in enumerate(folders)}

train_transforms = transforms.Compose([
                    PointSampler(1024),
                    Normalize(),
                    ToTensor()
                    ])

train_ds = PointCloudData(path, transforms=train_transforms)
valid_ds = PointCloudData(path, valid=True, folder='test', transforms=train_transforms)
inv_classes = {i: cat for cat, i in train_ds.classes.items()};
train_loader = DataLoader(dataset=train_ds, batch_size=32, num_workers=8, shuffle=True)
valid_loader = DataLoader(dataset=valid_ds, batch_size=64)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
pointnet = PointNetCls()
pointnet.to(device)

def train(model, train_loader, val_loader=None,  epochs=15):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data['pointcloud'].to(device).float(), data['category'].to(device)
            optimizer.zero_grad()
            outputs, m3x3, m64x64 = model(inputs.transpose(1,2))

            loss = pointnetloss(outputs, labels, m3x3, m64x64)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 10 == 9:    # print every 10 mini-batches
                    print('[Epoch: %d, Batch: %4d / %4d], loss: %.3f' %
                        (epoch + 1, i + 1, len(train_loader), running_loss / 10))
                    running_loss = 0.0

        model.eval()
        correct = total = 0

        # validation
        if val_loader:
            with torch.no_grad():
                for data in val_loader:
                    inputs, labels = data['pointcloud'].to(device).float(), data['category'].to(device)
                    outputs, __, __ = model(inputs.transpose(1,2))
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            val_acc = 100. * correct / total
            print('Valid accuracy: %d %%' % val_acc)

    return model
