from torchvision import transforms
import torch

trn_tfms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(5, (0.01,0.2), \
                            scale=(0.9,1.1)),
    transforms.Resize((100,100)),
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5),(0.5))
])

val_tfms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((100,100)),
    transforms.ToTensor(),
    transforms.Normalize((0.5),
                         (0.5),(0.5))
])


def save_model(model,optimizer,loss,epoch,name='model.pt'):
    """_summary_

    Args:
        model (pytorch model): Pytorch model to be saved
        optimizer (optim): Optimizer, for ex Adam
        loss: Loss value
        epoch: Number of epochs trained
        name (str, optional): Model name. Defaults to 'model.pt'.
    """    
    PATH = 'pretrained'
    torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'epoch': epoch,
            }, f'{PATH}/{name}')

def load_model(model,optimizer,PATH='pretrained/model.pt'):
    """_summary_

    Args:
        model (pytorch model): Pytorch model to load
        optimizer: Optimizer, for ex: Adam
        PATH (str, optional): Full path to the model file. Defaults to 'pretrained/model.pt'.

    Returns:
        model: pretrained model with state dict
        optimizer: saved value of optimizer
        loss: saved value of loss
    """
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    loss = checkpoint['loss']
    return model,optimizer,loss