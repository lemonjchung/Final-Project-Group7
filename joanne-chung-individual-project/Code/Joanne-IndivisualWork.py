# =================================================================
# ======================= Joanne Chung: individual work
# =================================================================

# ----------- 1. Read HDF5 files and make train_loader for PyTorch
class DatasetFromHdf5(torch.utils.data.Dataset):
    def __init__(self, file_path):
        super(DatasetFromHdf5, self).__init__()
        hf = h5py.File(file_path)
        self.data = hf.get('images')
        self.target = hf.get('labels')
        self.classes = hf.get('categories')

    def __getitem__(self, index):
        return torch.from_numpy(self.data[index, :, :, :].T).float(), self.target[index]

    def __len__(self):
        return self.data.shape[0]
        
# create datasets and data loaders
train_dataset = DatasetFromHdf5(h5train_file)
test_dataset = DatasetFromHdf5(h5test_file)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 13 - 4 
# ------
# 13 + 5 


# ----------- 2. Show sample images
# read in image data
image_data = h5py.File("food_train.h5", "r")
print('training database file:', image_data.filename)

# get labels and images for training data
labels = image_data.get('labels')
images = image_data.get('images')
target_classes = list(target_class.decode() for target_class in image_data.get("categories"))
print('')
print('image labels:', target_classes)
print('input shape:', images.shape)
print('labels shape:', labels.shape)

# show first 20 images
samplesize = 20
idx = slice(0, samplesize)
sample_labels = list(target_classes[x] for x in labels[idx])
sample_images = images[idx]
fig, m_x = plt.subplots(4, 5, figsize = (12, 12))
for ax, i in zip(m_x.flatten(), range(samplesize)):
    ax.imshow(sample_images[i])
    ax.set_title(sample_labels[i])
    ax.axis('off')
    # print(sample_labels[i])
    # print(sample_images[i])
    # print(labelnames[np.argmax(sample_labels[i])])
plt.show()


# 8 - 3 
# -----
# 8 + 10 


# ----------- 3. Find a best CNN model
# define the network model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(                # input size = 128 x 128
            nn.Conv2d(3, 64, kernel_size=10, padding=4),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, padding=1))  # output size = 64 x 64
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=10, padding=4),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, padding=1))  # output size = 32 x 32
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=10, padding=4),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, padding=1))  # output size = 16 x 16
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=10, padding=4),
            nn.BatchNorm2d(384),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, padding=1))  # output size = 8 x 8
        self.layer5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=4, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, padding=1))  # output size = 5 x 5
        self.layer6 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2))             # output size = 2 x 2
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(2 * 2 * 128, 12)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.dropout(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out






