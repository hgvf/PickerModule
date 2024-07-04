import torch
import torch.nn as nn
import torch.nn.functional as F

class PhaseNet(nn.Module):
    def __init__(self):
        super(PhaseNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=3, out_channels=8, kernel_size=1, stride=1)
        self.conv2 = nn.Conv1d(in_channels=8, out_channels=11, kernel_size=1, stride=4)
        self.conv3 = nn.Conv1d(in_channels=11, out_channels=11, kernel_size=1, stride=1)
        self.conv4 = nn.Conv1d(in_channels=11, out_channels=16, kernel_size=1, stride=4)
        self.conv5 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=1, stride=1)
        self.conv6 = nn.Conv1d(in_channels=16, out_channels=22, kernel_size=1, stride=4)
        self.conv7 = nn.Conv1d(in_channels=22, out_channels=22, kernel_size=1, stride=1)
        self.conv8 = nn.Conv1d(in_channels=22, out_channels=32, kernel_size=1, stride=4)
        self.conv9 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=1, stride=1)
        self.up1 = nn.ConvTranspose1d(in_channels=32, out_channels=22, kernel_size=3, stride=4)
        self.up2 = nn.ConvTranspose1d(in_channels=44, out_channels=16, kernel_size=4, stride=4)
        self.up3 = nn.ConvTranspose1d(in_channels=32, out_channels=11, kernel_size=3, stride=4)
        self.up4 = nn.ConvTranspose1d(in_channels=22, out_channels=8, kernel_size=1, stride=4)
        self.up5 = nn.Conv1d(in_channels=16, out_channels=3, kernel_size=1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x2 = self.conv2(x1)
        # print(x2.shape)
        x3 = self.relu(self.conv3(x2))
        x4 = self.conv4(x3)
        x5 = self.relu(self.conv5(x4))
        # print(x5.shape)
        x6 = self.conv6(x5)
        x7 = self.relu(self.conv7(x6))
        # print(x7.shape)
        x8 = self.conv8(x7)
        x9 = self.relu(self.conv9(x8))
        x10 = self.up1(x9)
        # print("up",x10.shape)
        x11 = self.relu(torch.cat([x10, x7], dim=1))
        x12 = self.up2(x11)
        # print(x12.shape)
        x13 = self.relu(torch.cat([x12, x5], dim=1))
        x14 = self.up3(x13)
        x15 = self.relu(torch.cat([x14, x3], dim=1))
        x16 = self.up4(x15)
        x17 = self.relu(torch.cat([x16, x1], dim=1))
        x18 = self.up5(x17)
        x18 = F.softmax(x18, dim=1)

        return x18
# if __name__ == "__main__":
#     inputs = torch.randn(1,3,3001)
#     print("輸入數據的维度：", inputs.shape)
#     model = PhaseNet()
#     model.eval()

#     # 將輸入數據傳遞給模型以獲取輸出
#     with torch.no_grad():
#         output = model(inputs)

#     print("輸出數據的維度：", output.shape)
