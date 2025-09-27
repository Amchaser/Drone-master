from environment import GridEnvironment
from q_learning import QLearningAgent
from sarsa import SarsaAgent
from policy_gradient import PolicyGradientAgent
from dqn import DQNAgent
import matplotlib.pyplot as plt
import numpy as np

def main():
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
    obstacles.append((0, 4, 1, 2))   # 下左
    obstacles.append((0, 14, 1, 2))  # 下右
    
    # 下边界障碍物 (避开终点附近区域)
    obstacles.append((19, 2, 1, 2))  # 上左
    obstacles.append((19, 14, 1, 2)) # 上右
    
    # 左边界障碍物 (避开起点附近区域)
    obstacles.append((2, 0, 2, 1))   # 左边界下
    obstacles.append((14, 0, 2, 1))  # 左边界上
    
    # 右边界障碍物 (避开终点附近区域)
    obstacles.append((2, 19, 2, 1))  # 右边界下侧
    obstacles.append((14, 19, 2, 1)) # 右边界上侧
    
    # 添加障碍物到环境中
    for obs in obstacles:
        env.add_obstacle(*obs)
    
    # 添加固定轨迹往复运动的动态障碍物
    env.add_dynamic_obstacles(2)
    
    # 测试不同奖励函数的性能
    reward_versions = ['v2', 'v4', 'v6']  # 使用我们新增的v6奖励函数进行测试
    results = {}
    
    for reward_version in reward_versions:
        print(f"\n使用奖励函数 {reward_version} 训练所有算法...")
        
        # 为每个奖励函数版本重新创建环境（确保初始状态一致）
        test_env = GridEnvironment(size=20)
        for obs in obstacles:
            test_env.add_obstacle(*obs)
        test_env.add_dynamic_obstacles(2)
        
        # 创建四个智能体（Q-learning、SARSA、策略梯度和DQN）
        q_agent = QLearningAgent(test_env, reward_version=reward_version)
        sarsa_agent = SarsaAgent(test_env, reward_version=reward_version)
        pg_agent = PolicyGradientAgent(test_env, reward_version=reward_version)
        dqn_agent = DQNAgent(test_env, reward_version=reward_version)
        
        # 训练智能体
        print(f"Training Q-Learning agent with reward {reward_version}...")
        q_rewards = q_agent.train(episodes=1000)
        
        print(f"Training SARSA agent with reward {reward_version}...")
        sarsa_rewards = sarsa_agent.train(episodes=1000)
        
        print(f"Training Policy Gradient agent with reward {reward_version}...")
        pg_rewards = pg_agent.train(episodes=1000)
        
        print(f"Training DQN agent with reward {reward_version}...")
        dqn_rewards = dqn_agent.train(episodes=1000)
        
        # 保存结果
        results[reward_version] = {
            'q_learning': q_rewards,
            'sarsa': sarsa_rewards,
            'policy_gradient': pg_rewards,
            'dqn': dqn_rewards
        }
    
    # 可视化不同奖励函数的性能对比
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    algorithms = ['q_learning', 'sarsa', 'policy_gradient', 'dqn']
    algorithm_names = ['Q-Learning', 'SARSA', 'Policy Gradient', 'DQN']
    
    for idx, (algorithm, name) in enumerate(zip(algorithms, algorithm_names)):
        ax = axes[idx]
        for reward_version in reward_versions:
            rewards = results[reward_version][algorithm]
            # 计算滑动平均以平滑曲线
            window_size = 50
            smoothed_rewards = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
            ax.plot(smoothed_rewards, label=f'{reward_version}')
        ax.set_title(f'{name} Performance with Different Reward Functions')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('reward_function_comparison.png')
    plt.show()
    
    # 使用v6奖励函数找到最终路径
    print("\n使用v6奖励函数生成最终路径...")
    env_final = GridEnvironment(size=20)
    for obs in obstacles:
        env_final.add_obstacle(*obs)
    env_final.add_dynamic_obstacles(2)
    
    # 创建智能体
    q_agent = QLearningAgent(env_final, reward_version='v6')
    sarsa_agent = SarsaAgent(env_final, reward_version='v6')
    pg_agent = PolicyGradientAgent(env_final, reward_version='v6')
    dqn_agent = DQNAgent(env_final, reward_version='v6')
    
    # 训练智能体
    print("Training Q-Learning agent...")
    q_agent.train(episodes=1000)
    
    print("Training SARSA agent...")
    sarsa_agent.train(episodes=1000)
    
    print("Training Policy Gradient agent...")
    pg_agent.train(episodes=1000)
    
    print("Training DQN agent...")
    dqn_agent.train(episodes=1000)
    
    # 找到路径（在查找路径前更新动态障碍物位置）
    env_final.update_dynamic_obstacles()  # 确保我们看到的是当前的动态障碍物位置
    q_path = q_agent.find_path()
    sarsa_path = sarsa_agent.find_path()
    pg_path = pg_agent.find_path()
    dqn_path = dqn_agent.find_path()
    
    # 验证路径合法性
    q_valid = q_agent.validate_path(q_path)
    sarsa_valid = sarsa_agent.validate_path(sarsa_path)
    pg_valid = pg_agent.validate_path(pg_path)
    dqn_valid = dqn_agent.validate_path(dqn_path)
    
    print(f"Q-learning path valid: {q_valid}")
    print(f"SARSA path valid: {sarsa_valid}")
    print(f"Policy Gradient path valid: {pg_valid}")
    print(f"DQN path valid: {dqn_valid}")
    
    # 可视化结果 - 对比四种算法的路径
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    axes = axes.flatten()
    
    # Q-learning路径
    ax1 = axes[0]
    ax1.imshow(env_final.grid, cmap='binary', origin='lower')
    ax1.plot(env_final.start[1], env_final.start[0], 'go', markersize=12, label='Start')
    ax1.plot(env_final.goal[1], env_final.goal[0], 'r*', markersize=15, label='Goal')
    
    # 绘制静态障碍物（包括边界）
    for (row, col, h, w) in env_final.obstacles:
        rect = plt.Rectangle((col - 0.5, row - 0.5), w, h, linewidth=1, 
                           edgecolor='gray', facecolor='gray', alpha=0.7)
        ax1.add_patch(rect)
    
    # 绘制动态障碍物轨迹
    for obstacle in env_final.dynamic_obstacles:
        trajectory_x = [pos[1] for pos in obstacle['trajectory']]
        trajectory_y = [pos[0] for pos in obstacle['trajectory']]
        ax1.plot(trajectory_x, trajectory_y, 'c--', alpha=0.5, linewidth=1)
    
    # 标记当前动态障碍物位置
    dynamic_positions = env_final.get_dynamic_obstacle_positions()
    if dynamic_positions:
        dyn_x = [pos[1] for pos in dynamic_positions]
        dyn_y = [pos[0] for pos in dynamic_positions]
        ax1.scatter(dyn_x, dyn_y, c='red', s=50, marker='s', label='Dynamic Obstacles')
    
    # 标记路径
    if q_path and q_valid:
        path_x = [p[1] for p in q_path]
        path_y = [p[0] for p in q_path]
        ax1.plot(path_x, path_y, 'b-', linewidth=2, label='Q-Learning Path')
    ax1.set_title('Q-Learning Path')
    ax1.legend()
    ax1.grid(True)
    
    # SARSA路径
    ax2 = axes[1]
    ax2.imshow(env_final.grid, cmap='binary', origin='lower')
    ax2.plot(env_final.start[1], env_final.start[0], 'go', markersize=12, label='Start')
    ax2.plot(env_final.goal[1], env_final.goal[0], 'r*', markersize=15, label='Goal')
    
    # 绘制静态障碍物
    for (row, col, h, w) in env_final.obstacles:
        rect = plt.Rectangle((col - 0.5, row - 0.5), w, h, linewidth=1, 
                           edgecolor='gray', facecolor='gray', alpha=0.7)
        ax2.add_patch(rect)
    
    # 绘制动态障碍物轨迹
    for obstacle in env_final.dynamic_obstacles:
        trajectory_x = [pos[1] for pos in obstacle['trajectory']]
        trajectory_y = [pos[0] for pos in obstacle['trajectory']]
        ax2.plot(trajectory_x, trajectory_y, 'c--', alpha=0.5, linewidth=1)
    
    # 标记当前动态障碍物位置
    if dynamic_positions:
        ax2.scatter(dyn_x, dyn_y, c='red', s=50, marker='s', label='Dynamic Obstacles')
    
    # 标记路径
    if sarsa_path and sarsa_valid:
        path_x = [p[1] for p in sarsa_path]
        path_y = [p[0] for p in sarsa_path]
        ax2.plot(path_x, path_y, 'g-', linewidth=2, label='SARSA Path')
    ax2.set_title('SARSA Path')
    ax2.legend()
    ax2.grid(True)
    
    # Policy Gradient路径
    ax3 = axes[2]
    ax3.imshow(env_final.grid, cmap='binary', origin='lower')
    ax3.plot(env_final.start[1], env_final.start[0], 'go', markersize=12, label='Start')
    ax3.plot(env_final.goal[1], env_final.goal[0], 'r*', markersize=15, label='Goal')
    
    # 绘制静态障碍物
    for (row, col, h, w) in env_final.obstacles:
        rect = plt.Rectangle((col - 0.5, row - 0.5), w, h, linewidth=1, 
                           edgecolor='gray', facecolor='gray', alpha=0.7)
        ax3.add_patch(rect)
    
    # 绘制动态障碍物轨迹
    for obstacle in env_final.dynamic_obstacles:
        trajectory_x = [pos[1] for pos in obstacle['trajectory']]
        trajectory_y = [pos[0] for pos in obstacle['trajectory']]
        ax3.plot(trajectory_x, trajectory_y, 'c--', alpha=0.5, linewidth=1)
    
    # 标记当前动态障碍物位置
    if dynamic_positions:
        ax3.scatter(dyn_x, dyn_y, c='red', s=50, marker='s', label='Dynamic Obstacles')
    
    # 标记路径
    if pg_path and pg_valid:
        path_x = [p[1] for p in pg_path]
        path_y = [p[0] for p in pg_path]
        ax3.plot(path_x, path_y, 'm-', linewidth=2, label='Policy Gradient Path')
    ax3.set_title('Policy Gradient Path')
    ax3.legend()
    ax3.grid(True)
    
    # DQN路径
    ax4 = axes[3]
    ax4.imshow(env_final.grid, cmap='binary', origin='lower')
    ax4.plot(env_final.start[1], env_final.start[0], 'go', markersize=12, label='Start')
    ax4.plot(env_final.goal[1], env_final.goal[0], 'r*', markersize=15, label='Goal')
    
    # 绘制静态障碍物
    for (row, col, h, w) in env_final.obstacles:
        rect = plt.Rectangle((col - 0.5, row - 0.5), w, h, linewidth=1, 
                           edgecolor='gray', facecolor='gray', alpha=0.7)
        ax4.add_patch(rect)
    
    # 绘制动态障碍物轨迹
    for obstacle in env_final.dynamic_obstacles:
        trajectory_x = [pos[1] for pos in obstacle['trajectory']]
        trajectory_y = [pos[0] for pos in obstacle['trajectory']]
        ax4.plot(trajectory_x, trajectory_y, 'c--', alpha=0.5, linewidth=1)
    
    # 标记当前动态障碍物位置
    if dynamic_positions:
        ax4.scatter(dyn_x, dyn_y, c='red', s=50, marker='s', label='Dynamic Obstacles')
    
    # 标记路径
    if dqn_path and dqn_valid:
        path_x = [p[1] for p in dqn_path]
        path_y = [p[0] for p in dqn_path]
        ax4.plot(path_x, path_y, 'orange', linewidth=2, label='DQN Path')
    ax4.set_title('DQN Path')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig('path_comparison.png')
    plt.show()
    
    # 输出性能统计
    print("\n=== 性能统计 ===")
    for reward_version in reward_versions:
        print(f"\n奖励函数 {reward_version}:")
        for algorithm in algorithms:
            rewards = results[reward_version][algorithm]
            avg_reward = np.mean(rewards[-100:])  # 最后100次episode的平均奖励
            max_reward = np.max(rewards)  # 最大奖励
            print(f"  {algorithm}: 平均奖励 = {avg_reward:.2f}, 最大奖励 = {max_reward:.2f}")

if __name__ == "__main__":
    main()