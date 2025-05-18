import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

print("PyTorch 版本:", torch.__version__)
print("CUDA 是否可用:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("使用的 GPU:", torch.cuda.get_device_name(0))


class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        """
        初始化線性神經網路。
        參數:
        - input_size: 輸入層大小
        - hidden_size: 隱藏層大小
        - output_size: 輸出層大小
        """
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        前向傳播。
        參數:
        - x: 輸入張量
        返回:
        - 輸出張量
        """
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

    def save(self, file_name='model.pth'):
        """
        儲存模型到指定檔案。
        參數:
        - file_name: 儲存的檔案名稱
        """
        model_folder_path = './model'
        os.makedirs(model_folder_path, exist_ok=True)  # 確保目錄存在
        file_path = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_path)
        print(f"模型已儲存至 {file_path}")


class QTrainer:
    def __init__(self, model, lr, gamma):
        """
        初始化 Q-Learning 訓練器。
        參數:
        - model: 神經網路模型
        - lr: 學習率
        - gamma: 折扣因子
        """
        self.model = model
        self.lr = lr
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        """
        執行一次訓練步驟。
        參數:
        - state: 當前狀態
        - action: 採取的動作
        - reward: 獎勵
        - next_state: 下一狀態
        - done: 是否結束
        """

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # 將輸入轉換為張量並移動到同一設備
        state = torch.tensor(state, dtype=torch.float).to(device)
        next_state = torch.tensor(next_state, dtype=torch.float).to(device)
        action = torch.tensor(action, dtype=torch.long).to(device)
        reward = torch.tensor(reward, dtype=torch.float).to(device)

        # 如果是單一樣本，調整維度
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            action = action.unsqueeze(0)
            reward = reward.unsqueeze(0)
            done = (done, )

        # 預測當前狀態的 Q 值
        pred = self.model(state)

        # 計算目標 Q 值
        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            target[idx][torch.argmax(action[idx]).item()] = Q_new

        # 計算損失並執行反向傳播
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()

        print(f"訓練步驟完成，損失值: {loss.item()}")



