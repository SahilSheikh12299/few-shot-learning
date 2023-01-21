from torch_snippets import *

class SiameseNetworkDataset(Dataset):
    def __init__(self, folder, transform=None, \
                 should_invert=True):
        self.folder = folder
        self.items = Glob(f'{self.folder}/*/*')
        self.transform = transform
    def __len__(self):        
        return len(self.items)
    def __getitem__(self, ix):
        itemA = self.items[ix]
        #person = fname(parent(itemA))
        person = str(itemA).rsplit('\\',2)[1]
        same_person = randint(2)
        if same_person:
            itemB = choose(Glob(f'{self.folder}/{person}/*', \
                                silent=True))
        else:
            while True:
                itemB = choose(self.items)
                #if person != fname(parent(itemB)):
                if person != str(itemB).rsplit('\\',2)[1]:
                    break
        imgA = read(itemA)
        imgB = read(itemB)
        if self.transform:
            imgA = self.transform(imgA)
            imgB = self.transform(imgB)
            return imgA, imgB, np.array([1-same_person])