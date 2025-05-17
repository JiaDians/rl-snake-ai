import matplotlib.pyplot as plt
import matplotlib

matplotlib.rc('font', family='Microsoft JhengHei')

# 啟用交互模式
plt.ion()

def plot(scores, mean_scores):
    """
    繪製訓練過程中的分數和平均分數曲線。

    參數:
    - scores: 每場遊戲的分數列表
    - mean_scores: 平均分數列表
    """
    if not scores or not mean_scores:
        print("警告: scores 或 mean_scores 為空，無法繪製圖表。")
        return

    plt.clf()  # 清除當前圖表
    plt.title('訓練過程')
    plt.xlabel('遊戲次數')
    plt.ylabel('分數')

    # 繪製分數曲線
    plt.plot(scores, label='分數', color='blue')
    # 繪製平均分數曲線
    plt.plot(mean_scores, label='平均分數', color='orange')

    plt.ylim(ymin=0)  # 設定 y 軸最小值為 0

    # 在圖表上顯示最後一個分數和平均分數
    if scores:
        plt.text(len(scores) - 1, scores[-1], str(scores[-1]), color='blue')
    if mean_scores:
        plt.text(len(mean_scores) - 1, mean_scores[-1], str(mean_scores[-1]), color='orange')

    plt.legend()  # 顯示圖例
    plt.tight_layout()  # 自動調整圖表佈局
    plt.show(block=False)  # 顯示圖表但不阻塞程式
    plt.pause(0.1)  # 暫停以更新圖表
