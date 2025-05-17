import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

# 初始化 Pygame
pygame.init()
font = pygame.font.Font('arial.ttf', 25)

# 定義方向
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

# 定義點結構
Point = namedtuple('Point', 'x, y')

# 顏色常量
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

# 遊戲常量
BLOCK_SIZE = 20
SPEED = 40

class SnakeGameAI:
    def __init__(self, width=240, height=240):
        self.width = width
        self.height = height
        # 初始化顯示
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        """重置遊戲狀態"""
        self.direction = Direction.RIGHT
        self.head = Point(self.width // 2, self.height // 2)
        self.snake = [
            self.head,
            Point(self.head.x - BLOCK_SIZE, self.head.y),
            Point(self.head.x - 2 * BLOCK_SIZE, self.head.y)
        ]
        self.score = 0
        self.food = None
        self.frame_iteration = 0
        self._place_food()

    def _place_food(self):
        """隨機放置食物，避免與蛇重疊"""
        while True:
            x = random.randint(0, (self.width - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            y = random.randint(0, (self.height - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            self.food = Point(x, y)
            if self.food not in self.snake:
                break

    def play_step(self, action):
        """執行遊戲的一步"""
        self.frame_iteration += 1
        # 處理使用者輸入
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # 移動蛇
        self._move(action)
        self.snake.insert(0, self.head)

        # 檢查遊戲是否結束
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # 檢查是否吃到食物
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()

        # 更新 UI 和時鐘
        self._update_ui()
        self.clock.tick(SPEED)

        return reward, game_over, self.score

    def is_collision(self, point=None):
        """檢查是否發生碰撞"""
        if point is None:
            point = self.head
        # 撞到邊界
        if point.x > self.width - BLOCK_SIZE or point.x < 0 or point.y > self.height - BLOCK_SIZE or point.y < 0:
            return True
        # 撞到自己
        if point in self.snake[1:]:
            return True
        return False

    def _update_ui(self):
        """更新遊戲畫面"""
        self.display.fill(BLACK)

        for segment in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(segment.x, segment.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(segment.x + 4, segment.y + 4, 12, 12))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        score_text = font.render(f"Score: {self.score}", True, WHITE)
        self.display.blit(score_text, [0, 0])
        pygame.display.flip()

    def _move(self, action):
        """根據動作更新蛇的方向和位置"""
        # [直行, 右轉, 左轉]
        directions = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = directions.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_direction = directions[idx]  # 不變
        elif np.array_equal(action, [0, 1, 0]):
            new_direction = directions[(idx + 1) % 4]  # 右轉
        else:  # [0, 0, 1]
            new_direction = directions[(idx - 1) % 4]  # 左轉

        self.direction = new_direction

        x, y = self.head.x, self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)