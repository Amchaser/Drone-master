"""
奖励函数性能对比脚本
用于比较不同奖励函数对强化学习算法性能的影响
"""

import numpy as np
import matplotlib.pyplot as plt
from environment import GridEnvironment
from q_learning import QLearningAgent
from sarsa import SarsaAgent
from policy_gradient import PolicyGradientAgent
from dqn import DQNAgent
import time

def create_environment():
    """创建标准测试环境"""
    # 创建环境
    env = GridEnvironment(size=20)
    
    # 添加静态障碍物（至少10个，分散布置，减小部分障碍物面积）
    obstacles = [
        (3, 3, 3, 3),   # 左上区域 (减小面积 4x4 -> 3x3)
        (15, 3, 3, 3),  # 右上区域 (减小面积 4x4 -> 3x3)
        (3, 15, 3, 3),  # 左下区域 (减小面积 4x4 -> 3x3)
        (15, 15, 3, 3), # 右下区域 (减小面积 4x4 -> 3x3)
        (8, 8, 2, 2),   # 中心偏左 (减小面积 3x3 -> 2x2)
        (12, 8, 2, 2),  # 中心偏右 (减小面积 3x3 -> 2x2)
        (8, 12, 2, 2),  # 中心偏下 (减小面积 3x3 -> 2x2)
        (12, 12, 2, 2), # 中心偏上 (减小面积 3x3 -> 2x2)
        (5, 10, 2, 1),  # 左侧中部 (减小面积 2x2 -> 2x1)
        (15, 10, 2, 1)  # 右侧中部 (减小面积 2x2 -> 2x1)
    ]
    
    # 在边界添加额外的静态障碍物，但避开起点(0,0)和终点(19,19)附近区域
    # 上边界障碍物 (避开起点附近区域)
    obstacles.append((0, 14, 1, 2))  # 下右
    
    # 左边界障碍物 (避开起点附近区域)
    obstacles.append((14, 0, 2, 1))  # 左边界上
    
    # 添加障碍物到环境中
    for obs in obstacles:
        env.add_obstacle(*obs)
    
    # 添加固定轨迹往复运动的动态障碍物
    env.add_dynamic_obstacles(2)
    
    return env

def train_with_reward_function(algorithm, reward_version, episodes=500):
    """
    使用指定奖励函数训练智能体

    Args:
        algorithm: 算法名称 ('q_learning', 'sarsa', 'policy_gradient', 'dqn')
        reward_version: 奖励函数版本 ('v1', 'v2', 'v3', 'v4', 'v5')
        episodes: 训练回合数

    Returns:
        rewards: 奖励历史
        path_length: 最终路径长度
        training_time: 训练时间
    """
    print(f"正在使用 {algorithm} 算法和奖励函数 {reward_version} 训练...")

    # 创建环境
    env = create_environment()

    # 创建智能体
    if algorithm == 'q_learning':
        agent = QLearningAgent(env, reward_version=reward_version)
    elif algorithm == 'sarsa':
        agent = SarsaAgent(env, reward_version=reward_version)
    elif algorithm == 'policy_gradient':
        agent = PolicyGradientAgent(env, reward_version=reward_version)
    elif algorithm == 'dqn':
        agent = DQNAgent(env, reward_version=reward_version)
    else:
        raise ValueError(f"未知算法: {algorithm}")

    # 记录训练开始时间
    start_time = time.time()

    # 训练智能体
    rewards = agent.train(episodes=episodes)

    # 记录训练结束时间
    end_time = time.time()
    training_time = end_time - start_time

    # 找到路径
    env.update_dynamic_obstacles()
    path = agent.find_path()
    path_length = len(path) if path else float('inf')

    # 验证路径
    valid = agent.validate_path(path)
    if not valid:
        path_length = float('inf')  # 无效路径视为无限长

    print(f"{algorithm} 算法使用奖励函数 {reward_version} 训练完成")
    print(f"  训练时间: {training_time:.2f} 秒")
    print(f"  最终奖励: {rewards[-1]:.2f}")
    print(f"  平均奖励: {np.mean(rewards[-100:]):.2f}")
    print(f"  路径长度: {path_length}")
    print()

    return rewards, path_length, training_time, valid

def compare_reward_functions_for_algorithm(algorithm):
    """比较不同奖励函数对指定算法的性能"""
    # 奖励函数版本列表
    reward_versions = ['v1', 'v2', 'v3', 'v4', 'v5']

    # 存储结果
    results = {}

    # 训练回合数
    episodes = 500

    # 对每个奖励函数进行训练
    for version in reward_versions:
        try:
            rewards, path_length, training_time, valid = train_with_reward_function(algorithm, version, episodes)
            results[version] = {
                'rewards': rewards,
                'path_length': path_length,
                'training_time': training_time,
                'valid': valid
            }
        except Exception as e:
            print(f"算法 {algorithm} 使用奖励函数 {version} 训练出错: {e}")
            results[version] = None

    # 可视化结果
    visualize_algorithm_results(algorithm, results, episodes)

    return results

def compare_algorithms_with_reward_function(reward_version):
    """比较不同算法使用指定奖励函数的性能"""
    # 算法列表
    algorithms = ['q_learning', 'sarsa', 'policy_gradient', 'dqn']

    # 存储结果
    results = {}

    # 训练回合数
    episodes = 500

    # 对每个算法进行训练
    for algorithm in algorithms:
        try:
            rewards, path_length, training_time, valid = train_with_reward_function(algorithm, reward_version, episodes)
            results[algorithm] = {
                'rewards': rewards,
                'path_length': path_length,
                'training_time': training_time,
                'valid': valid
            }
        except Exception as e:
            print(f"算法 {algorithm} 使用奖励函数 {reward_version} 训练出错: {e}")
            results[algorithm] = None

    # 可视化结果
    visualize_algorithm_comparison(reward_version, results, episodes)

    return results

def visualize_algorithm_results(algorithm, results, episodes):
    """可视化算法在不同奖励函数下的结果"""
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    # 1. 奖励曲线对比
    ax1 = axes[0]
    window_size = 20
    for version, data in results.items():
        if data is not None:
            rewards = data['rewards']
            if len(rewards) >= window_size:
                rewards_smooth = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
                ax1.plot(rewards_smooth, label=f'Version {version}')
    ax1.set_title(f'{algorithm} algorithm - the comparison of reward')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('reward')
    ax1.legend()
    ax1.grid(True)

    # 2. 训练时间对比
    ax2 = axes[1]
    versions = []
    times = []
    for version, data in results.items():
        if data is not None:
            versions.append(f'Version {version}')
            times.append(data['training_time'])

    bars = ax2.bar(versions, times, color=['blue', 'orange', 'green', 'red', 'purple'])
    ax2.set_title(f'{algorithm} algorithm - the comparison of train time')
    ax2.set_ylabel('T(s)')
    ax2.set_xlabel('the version of reward_funtion')

    # 在柱状图上显示数值
    for bar, time_val in zip(bars, times):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{time_val:.1f}s', ha='center', va='bottom')

    # 3. 路径长度对比
    ax3 = axes[2]
    versions = []
    lengths = []
    for version, data in results.items():
        if data is not None:
            versions.append(f'Version {version}')
            length = data['path_length'] if data['path_length'] != float('inf') else 0
            lengths.append(length)

    bars = ax3.bar(versions, lengths, color=['blue', 'orange', 'green', 'red', 'purple'])
    ax3.set_title(f'{algorithm} algorithm - the comparison of length ')
    ax3.set_ylabel('length')
    ax3.set_xlabel('reward_function')

    # 在柱状图上显示数值
    for bar, length_val in zip(bars, lengths):
        if length_val == 0:
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    '无效', ha='center', va='bottom')
        else:
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(int(length_val)), ha='center', va='bottom')

    # 4. 最终平均奖励对比
    ax4 = axes[3]
    versions = []
    avg_rewards = []
    for version, data in results.items():
        if data is not None:
            versions.append(f'Version {version}')
            avg_rewards.append(np.mean(data['rewards'][-50:]))

    bars = ax4.bar(versions, avg_rewards, color=['blue', 'orange', 'green', 'red', 'purple'])
    ax4.set_title(f'{algorithm} algorithm - the comparison of average reward')
    ax4.set_ylabel('average reward')
    ax4.set_xlabel('the reward_function')

    # 在柱状图上显示数值
    for bar, reward_val in zip(bars, avg_rewards):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{reward_val:.1f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

def visualize_algorithm_comparison(reward_version, results, episodes):
    """可视化不同算法使用相同奖励函数的结果"""
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    # 1. 奖励曲线对比
    ax1 = axes[0]
    window_size = 20
    for algorithm, data in results.items():
        if data is not None:
            rewards = data['rewards']
            if len(rewards) >= window_size:
                rewards_smooth = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
                ax1.plot(rewards_smooth, label=algorithm)
    ax1.set_title(f'reward_function {reward_version} - the comparison 0f the reward')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('reward')
    ax1.legend()
    ax1.grid(True)

    # 2. 训练时间对比
    ax2 = axes[1]
    algorithms = []
    times = []
    for algorithm, data in results.items():
        if data is not None:
            algorithms.append(algorithm)
            times.append(data['training_time'])

    bars = ax2.bar(algorithms, times)
    ax2.set_title(f'reward_function {reward_version} - the comparison of the time')
    ax2.set_ylabel('T(s)')
    ax2.set_xlabel('method')

    # 在柱状图上显示数值
    for bar, time_val in zip(bars, times):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{time_val:.1f}s', ha='center', va='bottom')

    # 3. 路径长度对比
    ax3 = axes[2]
    algorithms = []
    lengths = []
    for algorithm, data in results.items():
        if data is not None:
            algorithms.append(algorithm)
            length = data['path_length'] if data['path_length'] != float('inf') else 0
            lengths.append(length)

    bars = ax3.bar(algorithms, lengths)
    ax3.set_title(f'reward_function {reward_version} - the comparison of length')
    ax3.set_ylabel('length')
    ax3.set_xlabel('method')

    # 在柱状图上显示数值
    for bar, length_val in zip(bars, lengths):
        if length_val == 0:
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    'uneffective', ha='center', va='bottom')
        else:
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(int(length_val)), ha='center', va='bottom')

    # 4. 最终平均奖励对比
    ax4 = axes[3]
    algorithms = []
    avg_rewards = []
    for algorithm, data in results.items():
        if data is not None:
            algorithms.append(algorithm)
            avg_rewards.append(np.mean(data['rewards'][-50:]))

    bars = ax4.bar(algorithms, avg_rewards)
    ax4.set_title(f'reward_function {reward_version} - the comparison of the average reward')
    ax4.set_ylabel('the average reward')
    ax4.set_xlabel('method')

    # 在柱状图上显示数值
    for bar, reward_val in zip(bars, avg_rewards):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{reward_val:.1f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

def print_algorithm_summary(algorithm, results):
    """打印算法性能总结"""
    print("=" * 50)
    print(f"{algorithm} 算法奖励函数性能总结")
    print("=" * 50)

    for version, data in results.items():
        if data is not None:
            print(f"奖励函数 {version}:")
            print(f"  训练时间: {data['training_time']:.2f} 秒")
            print(f"  最终平均奖励: {np.mean(data['rewards'][-50:]):.2f}")
            if data['path_length'] != float('inf'):
                print(f"  路径长度: {data['path_length']}")
            else:
                print("  路径长度: 无效路径")
            print(f"  路径有效性: {'有效' if data['valid'] else '无效'}")
            print()

def visualize_path(env, path, title="Planned Path"):
    """可视化环境中的路径"""
    grid = np.zeros((env.size, env.size))

    # 标记障碍物
    for (x, y, w, h) in env.obstacles:
        for i in range(x, x + w):
            for j in range(y, y + h):
                if 0 <= i < env.size and 0 <= j < env.size:
                    grid[j, i] = -1  # 静态障碍物

    # 动态障碍物（当前时刻）
    for (x, y, w, h) in env.dynamic_obstacles:
        for i in range(x, x + w):
            for j in range(y, y + h):
                if 0 <= i < env.size and 0 <= j < env.size:
                    grid[j, i] = -1  # 用相同颜色表示

    # 路径
    for (x, y) in path:
        if (x, y, 1, 1) not in [(obs[0], obs[1], obs[2], obs[3]) for obs in env.obstacles + env.dynamic_obstacles]:
            grid[y, x] = 1

    plt.figure(figsize=(8, 8))
    cmap_colors = ['white', 'green', 'red']  # 分别代表空地、路径、障碍物
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(cmap_colors)
    bounds = [-1.5, -0.5, 0.5, 1.5]
    norm = plt.Normalize(-1.5, 1.5)
    img = plt.imshow(grid, cmap=cmap, norm=norm, origin='lower')

    # 添加网格线
    plt.grid(which='both', color='black', linewidth=0.5)
    plt.xticks(np.arange(-0.5, env.size, 1), [])
    plt.yticks(np.arange(-0.5, env.size, 1), [])
    plt.title(title)

    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='white', edgecolor='black', label='Free Space'),
        Patch(facecolor='green', edgecolor='black', label='Path'),
        Patch(facecolor='red', edgecolor='black', label='Obstacle')
    ]
    plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.2, 1))

    plt.tight_layout()
    plt.show()


def print_comparison_summary(reward_version, results):
    """打印算法对比总结"""
    print("=" * 50)
    print(f"奖励函数 {reward_version} 算法性能对比总结")
    print("=" * 50)

    for algorithm, data in results.items():
        if data is not None:
            print(f"{algorithm}:")
            print(f"  训练时间: {data['training_time']:.2f} 秒")
            print(f"  最终平均奖励: {np.mean(data['rewards'][-50:]):.2f}")
            if data['path_length'] != float('inf'):
                print(f"  路径长度: {data['path_length']}")
            else:
                print("  路径长度: 无效路径")
            print(f"  路径有效性: {'有效' if data['valid'] else '无效'}")
            print()

def run_single_test():
    """运行单个算法和奖励函数测试"""
    print("运行单个算法和奖励函数测试...")
    algorithm = input("请输入算法名称 (q_learning, sarsa, policy_gradient, dqn): ")
    reward_version = input("请输入奖励函数版本 (v1, v2, v3, v4, v5): ")

    rewards, path_length, training_time, valid = train_with_reward_function(algorithm, reward_version, episodes=500)

    # 重新创建环境以获取最终状态
    env = create_environment()
    if algorithm == 'q_learning':
        agent = QLearningAgent(env, reward_version=reward_version)
    elif algorithm == 'sarsa':
        agent = SarsaAgent(env, reward_version=reward_version)
    elif algorithm == 'policy_gradient':
        agent = PolicyGradientAgent(env, reward_version=reward_version)
    elif algorithm == 'dqn':
        agent = DQNAgent(env, reward_version=reward_version)
    else:
        raise ValueError(f"未知算法: {algorithm}")

    # 再次训练用于路径提取（或可保存模型）
    agent.train(episodes=500)
    env.update_dynamic_obstacles()  # 更新动态障碍物状态
    path = agent.find_path()

    # 可视化奖励曲线
    plt.figure(figsize=(10, 6))
    window_size = 20
    if len(rewards) >= window_size:
        rewards_smooth = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        plt.plot(rewards_smooth, label=f'{algorithm} - 奖励函数 {reward_version}')
    
    plt.title('the reward of one train')
    plt.xlabel('Episode')
    plt.ylabel('reward')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 可视化路径（仅当路径存在且有效）
    if path and valid:
        visualize_path(env, path, title=f"{algorithm} with Reward Function {reward_version}")
    else:
        print("无法可视化路径：路径无效或为空")

def run_full_comparison():
    """运行完整的奖励函数对比测试"""
    print("运行完整奖励函数对比测试...")
    algorithm = input("请输入算法名称 (q_learning, sarsa, policy_gradient, dqn): ")
    results = compare_reward_functions_for_algorithm(algorithm)
    print_algorithm_summary(algorithm, results)

def run_algorithm_comparison():
    """运行算法对比测试"""
    print("运行算法对比测试...")
    reward_version = input("请输入奖励函数版本 (v1, v2, v3, v4, v5): ")
    results = compare_algorithms_with_reward_function(reward_version)
    print_comparison_summary(reward_version, results)

if __name__ == "__main__":
    print("奖励函数性能对比工具")
    print("1. 运行完整奖励函数对比测试 (指定算法)")
    print("2. 运行算法对比测试 (指定奖励函数)")
    print("3. 运行单个测试")
    
    choice = input("请选择 (1, 2 或 3): ")
    
    if choice == "1":
        run_full_comparison()
    elif choice == "2":
        run_algorithm_comparison()
    elif choice == "3":
        run_single_test()
    else:
        print("无效选择，运行完整奖励函数对比测试")
        run_full_comparison()