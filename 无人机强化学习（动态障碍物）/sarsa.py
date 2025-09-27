import numpy as np
import random

class SarsaAgent:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.9, epsilon=0.1, reward_version='v1'):
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.q_table = np.zeros((env.size, env.size, 4))  # 4个动作：上、右、下、左
        
        # 导入奖励函数模块
        from reward_functions import get_reward_function
        self.reward_function = get_reward_function(reward_version)
        self.reward_version = reward_version
        
    def get_action(self, state):
        """选择动作"""
        if random.random() < self.epsilon:
            return random.randint(0, 3)  # 随机动作
        else:
            return np.argmax(self.q_table[state[0], state[1]])
    
    def update_q_value(self, state, action, reward, next_state, next_action):
        """更新Q值 (SARSA更新规则)"""
        current_q = self.q_table[state[0], state[1], action]
        next_q = self.q_table[next_state[0], next_state[1], next_action]
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * next_q - current_q)
        self.q_table[state[0], state[1], action] = new_q
    
    def get_next_state(self, state, action):
        """获取下一个状态"""
        x, y = state
        if action == 0:  # 上
            next_x, next_y = x - 1, y
        elif action == 1:  # 右
            next_x, next_y = x, y + 1
        elif action == 2:  # 下
            next_x, next_y = x + 1, y
        else:  # 左
            next_x, next_y = x, y - 1
            
        # 检查边界和障碍物
        if self.env.is_valid_position(next_x, next_y):
            return (next_x, next_y)
        else:
            return state  # 保持原位
    
    def calculate_reward(self, state, action, next_state):
        """
        计算奖励值
        
        Args:
            state: 当前状态
            action: 执行的动作
            next_state: 下一状态
            
        Returns:
            reward: 奖励值
        """
        goal_reached = (next_state == self.env.goal)
        collision = (state == next_state and next_state != self.env.goal)
        
        return self.reward_function(
            self.env, state, action, next_state, 
            goal_reached=goal_reached, 
            collision=collision
        )
    
    def train(self, episodes=1000):
        """训练智能体"""
        rewards_history = []
        
        for episode in range(episodes):
            # 每个episode开始时更新动态障碍物
            self.env.update_dynamic_obstacles()
            
            # 如果使用时间敏感奖励函数，需要重置episode_steps
            if self.reward_version == 'v5':
                self.env.episode_steps = 0
            
            state = self.env.start
            action = self.get_action(state)  # 初始动作
            total_reward = 0
            steps = 0
            
            while state != self.env.goal:
                next_state = self.get_next_state(state, action)
                
                # 使用新的奖励函数计算奖励
                reward = self.calculate_reward(state, action, next_state)
                
                # 选择下一个动作 (SARSA的关键区别)
                next_action = self.get_action(next_state)
                
                # 更新Q值
                self.update_q_value(state, action, reward, next_state, next_action)
                
                # 转移到下一个状态和动作
                state = next_state
                action = next_action
                
                total_reward += reward
                steps += 1
                
                # 如果使用时间敏感奖励函数，需要更新episode_steps
                if self.reward_version == 'v5':
                    self.env.episode_steps += 1
                
                if steps > 1000:  # 防止无限循环
                    break
            
            rewards_history.append(total_reward)
            
            # 每200个episode输出一次当前最优路径（使用贪婪策略找到的路径）
            # 修改为从第200个episode开始显示图像
            if episode % 200 == 0 and episode >= 200:
                print(f"Episode {episode}, Reward: {total_reward}")
                # 可视化当前最优路径（使用贪婪策略找到的路径）
                path = self.find_path()
                # 验证路径的合法性
                valid_path = self.validate_path(path)
                if valid_path:
                    self.env.visualize(path, f'SARSA Path - Episode {episode} - Reward {self.reward_version}')
        
        return rewards_history
    
    def validate_path(self, path):
        """验证路径是否合法（不穿越障碍物）"""
        if not path:
            return False
        
        # 检查起点和终点
        if not self.env.is_valid_position(path[0][0], path[0][1]) and path[0] != self.env.start:
            return False
        if not self.env.is_valid_position(path[-1][0], path[-1][1]) and path[-1] != self.env.goal:
            return False
        
        # 使用visited_states集合记录已访问状态，防止智能体陷入A→B→C→A类复杂循环
        visited_states = set()
        
        # 检查路径中相邻状态之间的移动是否合法
        for i in range(len(path) - 1):
            state = path[i]
            next_state = path[i + 1]
            
            # 检查当前位置是否是障碍物
            if not self.env.is_valid_position(state[0], state[1]) and state != self.env.start:
                return False
                
            # 检查下一个位置是否是障碍物
            if not self.env.is_valid_position(next_state[0], next_state[1]) and next_state != self.env.goal:
                return False
                
            # 检查从当前状态到下一个状态的移动是否合法
            action = None
            x, y = state
            nx, ny = next_state
            
            if nx == x - 1 and ny == y:  # 上
                action = 0
            elif nx == x and ny == y + 1:  # 右
                action = 1
            elif nx == x + 1 and ny == y:  # 下
                action = 2
            elif nx == x and ny == y - 1:  # 左
                action = 3
            else:
                # 如果移动不是四个基本方向之一，则非法
                return False
            
            # 检查该动作是否会导致碰撞或边界
            if not self.env.is_valid_position(nx, ny):
                return False
            
            # 检查是否形成循环
            if state in visited_states:
                return False
            visited_states.add(state)
        
        # 最后检查终点是否已访问
        if path[-1] in visited_states:
            return False
        
        return True
    
    def find_path(self):
        """找到从起点到终点的路径"""
        path = []
        state = self.env.start
        visited_states = set()
        
        # 添加路径合法性检查，考虑当前动态障碍物的位置
        while state != self.env.goal:
            # 获取最优动作
            action = np.argmax(self.q_table[state[0], state[1]])
            
            # 获取下一个状态
            next_state = self.get_next_state(state, action)
            
            # 检查是否形成循环
            if next_state in visited_states:
                # 尝试其他动作
                best_action = action
                for a in range(4):
                    test_next = self.get_next_state(state, a)
                    if (self.q_table[state[0], state[1], a] > self.q_table[state[0], state[1], best_action] and 
                        test_next not in visited_states):
                        best_action = a
                        next_state = test_next
                
                # 如果所有动作都会导致循环，就跳出
                if best_action == action and next_state in visited_states:
                    break
            
            # 检查下一步是否合法（不在障碍物中且在边界内）
            if not self.env.is_valid_position(next_state[0], next_state[1]):
                # 如果下一步不合法，尝试其他动作
                valid_actions = []
                for a in range(4):
                    test_next = self.get_next_state(state, a)
                    if self.env.is_valid_position(test_next[0], test_next[1]) and test_next not in visited_states:
                        valid_actions.append(a)
                
                # 如果有合法动作，选择Q值最大的
                if valid_actions:
                    best_action = max(valid_actions, key=lambda a: self.q_table[state[0], state[1], a])
                    next_state = self.get_next_state(state, best_action)
                else:
                    # 没有合法动作，终止搜索
                    break
            
            # 添加当前状态到路径
            path.append(state)
            visited_states.add(state)
            state = next_state
            
            # 防止无限循环
            if len(path) > 1000:
                break
        
        # 确保终点被包含
        if state == self.env.goal and self.env.is_valid_position(state[0], state[1]):
            path.append(state)
        
        return path

# 测试SARSA算法
if __name__ == "__main__":
    from environment import GridEnvironment
    import matplotlib.pyplot as plt
    
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
    #obstacles.append((0, 4, 1, 2))   # 下左
    obstacles.append((0, 14, 1, 2))  # 下右
    
    # 下边界障碍物 (避开终点附近区域)
    #obstacles.append((19, 2, 1, 2))  # 上左
    #obstacles.append((19, 14, 1, 2)) # 上右
    
    # 左边界障碍物 (避开起点附近区域)
    #obstacles.append((2, 0, 2, 1))   # 左边界下
    obstacles.append((14, 0, 2, 1))  # 左边界上
    
    # 右边界障碍物 (避开终点附近区域)
    #obstacles.append((2, 19, 2, 1))  # 右边界下侧
    #obstacles.append((14, 19, 2, 1)) # 右边界上侧
    
    # 添加障碍物到环境中
    for obs in obstacles:
        env.add_obstacle(*obs)
    
    # 添加固定轨迹往复运动的动态障碍物
    env.add_dynamic_obstacles(2)
    
    # 创建SARSA智能体
    sarsa_agent = SarsaAgent(env)
    
    # 训练智能体
    print("Training SARSA agent...")
    sarsa_rewards = sarsa_agent.train(episodes=1000)
    
    # 找到路径（在查找路径前更新动态障碍物位置）
    env.update_dynamic_obstacles()  # 确保我们看到的是当前的动态障碍物位置
    sarsa_path = sarsa_agent.find_path()
    
    # 验证路径合法性
    sarsa_valid = sarsa_agent.validate_path(sarsa_path)
    
    print(f"SARSA path valid: {sarsa_valid}")
    
    # 可视化结果
    if sarsa_path and sarsa_valid:
        env.visualize(sarsa_path, 'SARSA Path Planning')
    elif sarsa_path:
        print("Warning: SARSA generated an invalid path")
        env.visualize(sarsa_path, 'SARSA Path Planning (Invalid)')
    else:
        env.visualize([], 'SARSA Path Planning (No Path Found)')
    
    # 绘制奖励曲线
    plt.figure(figsize=(10, 5))
    
    # 计算滑动平均以更好地显示趋势
    window_size = 50
    sarsa_rewards_smooth = np.convolve(sarsa_rewards, np.ones(window_size)/window_size, mode='valid')
    
    episodes_range = np.arange(len(sarsa_rewards_smooth))
    plt.plot(episodes_range, sarsa_rewards_smooth, label='SARSA', color='blue', linewidth=2)
    plt.title('SARSA Cumulative Reward over Episodes', fontsize=14, fontweight='bold')
    plt.xlabel('Episode')
    plt.ylabel('Reward (Smoothed)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()
    
    # 输出统计信息
    print("\n=== SARSA Performance ===")
    print(f"SARSA - Path length: {len(sarsa_path)}")
    print(f"SARSA - Final reward: {sarsa_rewards[-1]}")
    print(f"SARSA - Average reward (last 100 episodes): {np.mean(sarsa_rewards[-100:]):.2f}")