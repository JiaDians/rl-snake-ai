import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot

# 常量
MAX_MEMORY = 100_000  # 最大記憶體大小
BATCH_SIZE = 1000  # 批次大小
LR = 0.001  # 學習率

class Agent:
    def __init__(self):
        self.n_games = 0  # 遊戲次數
        self.epsilon = 0  # 探索率
        self.gamma = 0.9  # 折扣率
        self.memory = deque(maxlen=MAX_MEMORY)  # 經驗回放緩衝區
        self.model = Linear_QNet(11, 256, 3)  # 神經網路模型
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)  # 訓練器

    def get_state(self, game):
        """獲取遊戲的當前狀態，並轉換為特徵向量。"""
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # 前方危險
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # 右方危險
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # 左方危險
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),

            # 移動方向
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # 食物位置
            game.food.x < game.head.x,  # 食物在左邊
            game.food.x > game.head.x,  # 食物在右邊
            game.food.y < game.head.y,  # 食物在上方
            game.food.y > game.head.y  # 食物在下方
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        """將經驗存入記憶體中。"""
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        """使用記憶體中的一批經驗訓練模型。"""
        mini_sample = random.sample(self.memory, BATCH_SIZE) if len(self.memory) > BATCH_SIZE else self.memory
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        """使用單次經驗訓練模型。"""
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        """使用 epsilon-greedy 策略決定下一步動作。"""
        self.epsilon = max(10, 80 - self.n_games)  # 隨著遊戲次數增加，減少探索率
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)  # 隨機選擇動作
            final_move[move] = 1
        else:
            state_tensor = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state_tensor)  # 使用模型預測
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
    """訓練智能體玩貪吃蛇遊戲。"""
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()

    while True:
        # 獲取當前狀態
        state_old = agent.get_state(game)

        # 獲取動作
        final_move = agent.get_action(state_old)

        # 執行動作並獲取新狀態
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # 訓練短期記憶
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # 記住經驗
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # 訓練長期記憶並重置遊戲
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print(f'遊戲 {agent.n_games}, 分數: {score}, 最高分: {record}')

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()