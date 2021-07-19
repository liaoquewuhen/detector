import torch 
import torch.nn as nn

class d1Net(nn.Module):
    def __init__(self):
        super(d1Net, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=10, kernel_size=3, stride=2),  #in:784*1*b, out:392*10*b
            nn.MaxPool1d(kernel_size=3, stride=2),                               #in:392*10*b, out:196*10*b
            nn.ReLU(),
            nn.Conv1d(10, 20, 3, 2),                                              #in:196*10*b, out:98*20*b
            nn.MaxPool1d(kernel_size=3, stride=2),                                #in:98*20*b, out:49*20*b
            nn.ReLU(),
            nn.Conv1d(20, 40, 4)                                                   #input:49*20*b, 12*40*b
        )
        self.fc = nn.Sequential(
            nn.Linear(1800, 120),
            nn.Linear(120,84),
            nn.Linear(84,10)
        )
        
    def forward(self,x):
        x = x.view(-1,1,784)
        #print(x.shape)
        x = self.conv(x)
        #print(x.shape)
        x = x.view(-1,1800)
        #print(x.shape)
        x = self.fc(x)
        #print(x.shape)
        return x

def load_model():
    net = d1Net()
    # net.load_state_dict(torch.load(r'D:\py\textDectectionDocker\mnist1d_detection_notzip.pkl'))
    net.load_state_dict(torch.load(r'mnist1d_detection_notzip.pkl'))
    #torch.save(net.state_dict(), "mnist1d_detection_notzip.pkl",_use_new_zipfile_serialization=False)
    return net

def detection(net,img):
    x = net(torch.from_numpy(img).float())
    #print(x)
    return torch.max(x,1).indices[0]