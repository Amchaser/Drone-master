import numpy as np
import matplotlib.pyplot as plt
import time
from q_learning import QLearningAgent
from sarsa import SarsaAgent
from policy_gradient import PolicyGradientAgent
from dqn import DQNAgent
from environment import GridEnvironment
import copy

# 设置中文字体以消除警告
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def run_reinforcement_learning_selected(env, selected_algorithms, reward_version='v2', episodes=300):
    """
    在给定环境中运行选定的强化学习算法
    
    参数:
    env: GridEnvironment对象
    selected_algorithms: 选择要运行的算法列表
    reward_version: 奖励函数版本
    episodes: 训练轮次 (DQN会使用较小的值以提高效率)
    
    返回:
    dict: 包含选定算法结果的字典
    """
    try:
        # 为每个选定的算法创建独立的环境副本
        env_copies = {}
        for i, algorithm in enumerate(selected_algorithms):
            if i == 0:
                env_copy = GridEnvironment(env.size)
                env_copy.grid = env.grid.copy()
                env_copy.start = env.start
                env_copy.goal = env.goal
                env_copy.obstacles = copy.deepcopy(env.obstacles)
                env_copy.dynamic_obstacles = copy.deepcopy(env.dynamic_obstacles)
            else:
                env_copy = copy.deepcopy(env_copies[selected_algorithms[0]])
            env_copies[algorithm] = env_copy

        # 创建选定的智能体
        agents = {}
        if 'q_learning' in selected_algorithms:
            agents['q_learning'] = QLearningAgent(env_copies['q_learning'], reward_version=reward_version)
        if 'sarsa' in selected_algorithms:
            agents['sarsa'] = SarsaAgent(env_copies['sarsa'], reward_version=reward_version)
        if 'policy_gradient' in selected_algorithms:
            agents['policy_gradient'] = PolicyGradientAgent(env_copies['policy_gradient'], reward_version=reward_version)
        if 'dqn' in selected_algorithms:
            agents['dqn'] = DQNAgent(env_copies['dqn'], reward_version=reward_version)

        # 训练智能体
        print(f"使用奖励函数 {reward_version} 训练智能体...")
        
        # 为DQN使用与其它算法相同的训练次数，移除硬编码的限制
        dqn_episodes = episodes  # 使用用户设置的训练次数
        
        results = {}
        
        start_time = time.time()
        
        # 训练选定的算法
        if 'q_learning' in selected_algorithms:
            q_rewards = agents['q_learning'].train(episodes=episodes)
            q_time = time.time() - start_time
            print("Q-Learning训练完成")
            results['q_learning'] = {
                'rewards': q_rewards,
                'training_time': q_time
            }

        if 'sarsa' in selected_algorithms:
            sarsa_start_time = time.time()
            sarsa_rewards = agents['sarsa'].train(episodes=episodes)
            sarsa_time = time.time() - sarsa_start_time
            print("SARSA训练完成")
            results['sarsa'] = {
                'rewards': sarsa_rewards,
                'training_time': sarsa_time
            }

        if 'policy_gradient' in selected_algorithms:
            pg_start_time = time.time()
            pg_rewards = agents['policy_gradient'].train(episodes=episodes)
            pg_time = time.time() - pg_start_time
            print("Policy Gradient训练完成")
            results['policy_gradient'] = {
                'rewards': pg_rewards,
                'training_time': pg_time
            }

        if 'dqn' in selected_algorithms:
            dqn_start_time = time.time()
            dqn_rewards = agents['dqn'].train(episodes=dqn_episodes)
            dqn_time = time.time() - dqn_start_time
            print("DQN训练完成")
            results['dqn'] = {
                'rewards': dqn_rewards,
                'training_time': dqn_time
            }
        
        # 更新动态障碍物位置并查找路径
        env.update_dynamic_obstacles()
        
        # 查找选定算法的路径
        if 'q_learning' in selected_algorithms:
            q_path = agents['q_learning'].find_path()
            q_valid = agents['q_learning'].validate_path(q_path)
            results['q_learning'].update({
                'path': q_path,
                'valid': q_valid
            })

        if 'sarsa' in selected_algorithms:
            sarsa_path = agents['sarsa'].find_path()
            sarsa_valid = agents['sarsa'].validate_path(sarsa_path)
            results['sarsa'].update({
                'path': sarsa_path,
                'valid': sarsa_valid
            })

        if 'policy_gradient' in selected_algorithms:
            pg_path = agents['policy_gradient'].find_path()
            pg_valid = agents['policy_gradient'].validate_path(pg_path)
            results['policy_gradient'].update({
                'path': pg_path,
                'valid': pg_valid
            })

        if 'dqn' in selected_algorithms:
            dqn_path = agents['dqn'].find_path()
            dqn_valid = agents['dqn'].validate_path(dqn_path)
            results['dqn'].update({
                'path': dqn_path,
                'valid': dqn_valid
            })
        
        print(f"DQN实际训练轮次: {dqn_episodes}/{episodes}")
        return results
        
    except Exception as e:
        print(f"运行强化学习算法时出错: {str(e)}")
        return None

def run_reinforcement_learning(env, reward_version='v2', episodes=300):
    """
    在给定环境中运行所有强化学习算法
    
    参数:
    env: GridEnvironment对象
    reward_version: 奖励函数版本
    episodes: 训练轮次 (DQN会使用较小的值以提高效率)
    
    返回:
    dict: 包含所有算法结果的字典
    """
    # 调用新的函数，传入所有算法
    return run_reinforcement_learning_selected(env, ['q_learning', 'sarsa', 'policy_gradient', 'dqn'], reward_version, episodes)

def visualize_results(env, results, colors, markers):
    """可视化选定算法的路径规划结果"""
    # 根据结果数量确定子图布局
    num_algorithms = len(results)
    if num_algorithms <= 1:
        fig, axes = plt.subplots(1, 1, figsize=(8, 8))
        axes = [axes] if num_algorithms > 0 else []
    elif num_algorithms == 2:
        fig, axes = plt.subplots(1, 2, figsize=(15, 8))
        axes = axes.flatten()
    elif num_algorithms <= 4:
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        axes = axes.flatten()
    else:
        # 如果算法数量超过4个，使用更多行
        rows = (num_algorithms + 1) // 2
        fig, axes = plt.subplots(rows, 2, figsize=(15, 8*rows))
        axes = axes.flatten()
    
    # 获取动态障碍物位置用于显示
    dynamic_positions = env.get_dynamic_obstacle_positions()
    
    # 算法名称映射
    algorithm_names = {
        'q_learning': 'Q-Learning',
        'sarsa': 'SARSA',
        'policy_gradient': 'Policy Gradient',
        'dqn': 'DQN'
    }
    
    # 颜色和标记映射
    color_map = {
        'q_learning': colors[0] if len(colors) > 0 else 'blue',
        'sarsa': colors[1] if len(colors) > 1 else 'green',
        'policy_gradient': colors[2] if len(colors) > 2 else 'red',
        'dqn': colors[3] if len(colors) > 3 else 'magenta'
    }
    
    marker_map = {
        'q_learning': markers[0] if len(markers) > 0 else 'o',
        'sarsa': markers[1] if len(markers) > 1 else 's',
        'policy_gradient': markers[2] if len(markers) > 2 else '^',
        'dqn': markers[3] if len(markers) > 3 else 'd'
    }
    
    for idx, (algorithm_key, result_data) in enumerate(results.items()):
        ax = axes[idx] if idx < len(axes) else None
        if ax is None:
            continue
            
        name = algorithm_names.get(algorithm_key, algorithm_key)
        path = result_data.get('path', [])
        valid = result_data.get('valid', False)
        
        # 绘制网格世界
        grid_display = env.grid.copy()
        # 将静态障碍物标记为-1（黑色）
        grid_display[grid_display == 1] = -1
        # 将起点标记为2（绿色）
        grid_display[env.start] = 2
        # 将终点标记为3（红色）
        grid_display[env.goal] = 3
        
        # 显示网格
        cmap = plt.cm.colors.ListedColormap(['white', 'black', 'green', 'red'])
        ax.imshow(grid_display, cmap=cmap, alpha=0.7)
        
        # 标记动态障碍物
        if dynamic_positions:
            dyn_x = [pos[1] for pos in dynamic_positions]
            dyn_y = [pos[0] for pos in dynamic_positions]
            ax.scatter(dyn_x, dyn_y, c='red', s=50, marker='s', label='动态障碍物')
        
        # 标记路径
        if path:
            # 修复坐标翻转问题，保持正确的x和y对应关系
            path_x = [p[1] for p in path]  # x坐标对应列
            path_y = [p[0] for p in path]  # y坐标对应行
            path_label = f'{name} 路径 (长度: {len(path)})'
            if not valid:
                path_label += ' [无效]'
            ax.plot(path_x, path_y, color=color_map[algorithm_key], linewidth=2, marker=marker_map[algorithm_key], markersize=4,
                   label=path_label)
        elif path is not None:
            ax.text(0.5, 0.5, f'{name} 未找到路径', transform=ax.transAxes, 
                   ha='center', va='center', fontsize=12, color='red')
        else:
            ax.text(0.5, 0.5, f'{name} 无路径数据', transform=ax.transAxes, 
                   ha='center', va='center', fontsize=12, color='gray')
        
        ax.set_title(f'{name} 路径规划结果')
        ax.legend()
        ax.grid(True)
    
    # 隐藏多余的子图
    for idx in range(len(results), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.show()
    
    plt.tight_layout()
    plt.show()

    # 绘制奖励曲线
    plt.figure(figsize=(10, 6))
    algorithm_names = ['Q-Learning', 'SARSA', 'Policy Gradient', 'DQN']
    colors = ['blue', 'green', 'red', 'magenta']
    
    for algorithm, name, color in zip(['q_learning', 'sarsa', 'policy_gradient', 'dqn'], 
                                    algorithm_names, colors):
        rewards = results[algorithm]['rewards']
        if rewards:
            # 计算滑动平均
            window_size = max(1, len(rewards) // 50)  # 窗口大小为总长度的1/50
            if window_size > 1:
                smoothed_rewards = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
                plt.plot(smoothed_rewards, label=name, color=color)
            else:
                plt.plot(rewards, label=name, color=color)
    
    plt.title('累积奖励随训练轮次变化')
    plt.xlabel('训练轮次')
    plt.ylabel('奖励')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # 绘制训练时间对比图
    plt.figure(figsize=(10, 6))
    algorithms = ['Q-Learning', 'SARSA', 'Policy Gradient', 'DQN']
    training_times = [
        results['q_learning']['training_time'],
        results['sarsa']['training_time'],
        results['policy_gradient']['training_time'],
        results['dqn']['training_time']
    ]
    
    bars = plt.bar(algorithms, training_times, color=['blue', 'green', 'red', 'magenta'])
    plt.title('训练耗时对比')
    plt.xlabel('算法')
    plt.ylabel('训练耗时 (秒)')
    
    # 在柱状图上显示数值
    for bar, time_val in zip(bars, training_times):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                f'{time_val:.2f}s', ha='center', va='bottom')
    
    plt.grid(True, axis='y')
    plt.show()