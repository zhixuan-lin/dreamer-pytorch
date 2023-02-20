import torch 
import torch.nn as nn

class ConvEncoder(nn.Module):
    def __init__(self, act: nn.Module = nn.ReLU(inplace=True)):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.act =  act
        self.fc = nn.Linear(64 * 64, 256)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.conv3(x)
        x = self.act(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class ConvDecoder(nn.Module):
    def __init__(self, act: nn.Module = nn.ReLU(inplace=True)):
        super().__init__()
        self.fc = nn.Linear(256, 64 * 64)
        self.deconv1 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(32, 3, kernel_size=8, stride=4, padding=2)
        self.act = act
        
    def forward(self, x):
        x = self.fc(x)
        x = self.act(x)
        x = x.view(x.size(0), 64, 8, 8)
        x = self.deconv1(x)
        x = self.act(x)
        x = self.deconv2(x)
        x = self.act(x)
        x = self.deconv3(x)
        return x
    
class ConvDecoder(nn.Module):
    def __init__(self, act: nn.Module = nn.ReLU(inplace=True)):
        super().__init__()
        self.fc = nn.Linear(256, 64 * 64)
        self.deconv1 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(32, 3, kernel_size=8, stride=4, padding=2)
        self.act = act
        
    def forward(self, x):
        x = self.fc(x)
        x = self.act(x)
        x = x.view(x.size(0), 64, 8, 8)
        x = self.deconv1(x)
        x = self.act(x)
        x = self.deconv2(x)
        x = self.act(x)
        x = self.deconv3(x)
        return x

class Actor(nn.Module):
    def __init__(self, action_shape,act: nn.Module = nn.ReLU(inplace=True)):
        super().__init__()
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, action_shape)
        self.act = act
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x
class Critic(nn.Module):
    def __init__(self,act: nn.Module = nn.ReLU(inplace=True)):
        super().__init__()
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 1)
        self.act = act
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

# Script version
if __name__ == '__main__':
    
    def test_encoder():
        encoder = ConvEncoder()
        x = torch.randn(1,3,64,64)
        z = encoder(x)
        print(z.shape)
        
    def test_decoder():
        decoder = ConvDecoder()
        z= torch.randn(1,256)
        x = decoder(z)
        print(x.shape)

    def test_actor():
        actor = Actor(4)
        z = torch.randn(1,256)
        a = actor(z)
        print(a.shape)
    
    def test_critic():
        critic = Critic()
        z = torch.randn(1,256)
        q = critic(z)
        print(q.shape)
        
    #test_encoder()
    #test_decoder()
    #test_actor()
    #test_critic()
    
