import torch
import torch.nn as nn





class Occ_Detector(torch.nn.Module):

    def __init__(self):
        super(Occ_Detector, self).__init__()
        
        keep_prob = 1
        # L1 ImgIn shape=(?, 64, 64, 1)
        # Conv -> (?, 64, 64, 32)
        # Pool -> (?, 32, 32, 32)
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(p=1 - keep_prob))
        # L2 ImgIn shape=(?, 32, 32, 32)
        # Conv      ->(?, 32, 32, 64)
        # Pool      ->(?, 16, 16, 64)
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(p=1 - keep_prob))
        # L3 ImgIn shape=(?, 16, 16, 64)
        # Conv ->(?, 16,16, 128)
        # Pool ->(?, 8, 8, 128)
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            torch.nn.Dropout(p=1 - keep_prob))

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        
        
        # L4 FC 4x4x128 inputs -> 625 outputs
        self.fc1 = torch.nn.Linear(128, 64, bias=True)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.layer4 = torch.nn.Sequential(
            self.fc1,
            torch.nn.ReLU(),
            torch.nn.Dropout(p=1 - keep_prob))
        # L5 Final FC 625 inputs -> 10 outputs
        self.fc2 = torch.nn.Linear(64, 3, bias=True)
        torch.nn.init.xavier_uniform_(self.fc2.weight) # initialize parameters
        
        
        self.fc_amount_head = torch.nn.Linear(64, 1, bias=True)
        torch.nn.init.xavier_uniform_(self.fc_amount_head.weight) # initialize parameters
        
        self.classification_head = torch.nn.Sequential(
            self.fc2, 
            torch.nn.Softmax(dim=1)
        )
        
        self.layer_amount = torch.nn.Sequential(
            self.fc_amount_head,
            torch.nn.Sigmoid()
        )
        
        return 
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        
        out = out.view(out.size(0), -1)   # Flatten them for FC
        out = self.layer4(out)
        
        logits = self.classification_head(out)
        occ_amount = self.layer_amount(out).squeeze()
        
        return logits, occ_amount      #(Batch, num_classes, 1)