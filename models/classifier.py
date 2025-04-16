import torch.nn as nn

class ViTLargeClassifier(nn.Module):
    def __init__(self, input_dim=1024, num_classes=2):
        super(ViTLargeClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 768),
            nn.BatchNorm1d(768),
            nn.GELU(), 
            nn.Dropout(0.5),
            
            nn.Linear(768, 384),
            nn.BatchNorm1d(384),
            nn.GELU(),
            nn.Dropout(0.4),
            
            nn.Linear(384, 192),
            nn.BatchNorm1d(192),
            nn.GELU(),
            nn.Dropout(0.3),
            
            nn.Linear(192, num_classes)
        )
        
        # 权重初始化
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.classifier(x)