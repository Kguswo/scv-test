import torch
import torch.nn as nn

# 모델 이름: model_5_v19
# 데이터셋: MNIST
# 학습 데이터 수: 50000
# 테스트 데이터 수: 10000
# 레이블 수: 10
# 에폭 수: 2

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer0 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, padding_mode='zeros', bias=True)
        self.layer1 = nn.ReLU()
        self.layer2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.layer3 = nn.Flatten(start_dim=1, end_dim=-1)
        self.layer4 = nn.Linear(in_features=5408, out_features=10, bias=True)

    def forward(self, x):
        # Input shape: [batch_size, channels, height, width]
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
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