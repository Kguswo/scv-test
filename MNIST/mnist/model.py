import torch
import torch.nn as nn

# 모델 이름: model_49_v85
# 데이터셋: MNIST
# 학습 데이터 수: 50000
# 테스트 데이터 수: 10000
# 레이블 수: 10
# 에폭 수: 2

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer0 = nn.Conv2d(in_channels=1, out_channels=24, kernel_size=(6, 6), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, padding_mode='zeros', bias=True)
        self.layer1 = nn.ReflectionPad2d()
        self.layer2 = nn.LeakyReLU(negative_slope=1.0)
        self.layer3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.layer4 = nn.Conv2d(in_channels=24, out_channels=48, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, padding_mode='zeros', bias=True)
        self.layer5 = nn.ReplicationPad2d()
        self.layer6 = nn.ELU(alpha=1.0)
        self.layer7 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.layer8 = nn.ConvTranspose2d(in_channels=48, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, padding_mode='zeros', bias=True)
        self.layer9 = nn.GELU()
        self.layer10 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.layer11 = nn.Flatten(start_dim=1, end_dim=-1)
        self.layer12 = nn.Linear(in_features=512, out_features=256, bias=True)
        self.layer13 = nn.LeakyReLU(negative_slope=0.2)
        self.layer14 = nn.Linear(in_features=256, out_features=128, bias=True)
        self.layer15 = nn.ELU(alpha=1.0)
        self.layer16 = nn.Linear(in_features=128, out_features=10, bias=True)

    def forward(self, x):
        # Input shape: [batch_size, channels, height, width]
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.layer11(x)
        x = self.layer12(x)
        x = self.layer13(x)
        x = self.layer14(x)
        x = self.layer15(x)
        x = self.layer16(x)
        return x

if __name__ == '__main__':
    # 모델 인스턴스 생성
    model = Model()
    print('모델 구조:')
    print(model)
    
    # 입력 텐서 예제
    batch_size = 1  # 배치 크기
    channels = 1  # 입력 채널 수
    height = 28  # 입력 높이
    width = 28  # 입력 너비
    x = torch.randn(batch_size, channels, height, width)
    
    # 순전파 실행
    output = model(x)
    print(f'입력 shape: {x.shape}')
    print(f'출력 shape: {output.shape}')