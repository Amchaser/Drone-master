import numpy as np
import random

class PolicyGradientAgent:
    def __init__(self, env, learning_rate=0.01, discount_factor=0.99, epsilon=0.8, epsilon_decay=0.995, epsilon_min=0.1, reward_version='v1'):
        # 修改：调整超参数以提高性能
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # 策略网络（使用简单的线性函数近似）
        # 修改：使用更好的权重初始化
        self.weights = np.random.randn(env.size * env.size, 4) * 0.1
        
        # 存储episode的经验
        self.states = []
        self.actions = []
        self.rewards = []
        
        # 添加基线以减少方差
        self.baseline = 0
        
        # 导入奖励函数模块
        from reward_functions import get_reward_function
        self.reward_function = get_reward_function(reward_version)
        self.reward_version = reward_version
        
    def get_state_index(self, position):
        """将位置转换为状态索引"""
        return position[0] * self.env.size + position[1]
    
    def get_action(self, state):
        """根据策略选择动作"""
        # 添加探索机制 - ε-greedy策略
        if random.random() < self.epsilon:
            return random.randint(0, 3)  # 随机动作
        else:
            state_idx = self.get_state_index(state)
            logits = self.weights[state_idx]
            # 计算softmax概率
            exp_logits = np.exp(logits - np.max(logits))  # 数值稳定性
            probs = exp_logits / np.sum(exp_logits)
            
            # 根据概率选择动作
            return np.random.choice(4, p=probs)
    
    def get_action_probs(self, state):
        """获取动作概率分布"""
        state_idx = self.get_state_index(state)
        logits = self.weights[state_idx]
        # 计算softmax概率
        exp_logits = np.exp(logits - np.max(logits))  # 数值稳定性
        probs = exp_logits / np.sum(exp_logits)
        return probs
    
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
    
    def store_experience(self, state, action, reward):
        """存储经验"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
    
    def discount_rewards(self, rewards):
        """计算折扣奖励"""
        discounted = np.zeros_like(rewards, dtype=float)
        running_add = 0
        for t in reversed(range(len(rewards))):
            running_add = running_add * self.discount_factor + rewards[t]
            discounted[t] = running_add
        return discounted
    
    def update_policy(self):
        """更新策略"""
        if len(self.rewards) == 0:
            return
            
        # 计算折扣奖励
        discounted_rewards = self.discount_rewards(self.rewards)
        
        # 更新基线（使用移动平均）
        self.baseline = 0.9 * self.baseline + 0.1 * np.mean(discounted_rewards)
        
        # 标准化奖励以提高稳定性
        if np.std(discounted_rewards) > 0:
            discounted_rewards = (discounted_rewards - self.baseline) / (np.std(discounted_rewards) + 1e-9)
        
        # 更新策略
        for i in range(len(self.states)):
            state_idx = self.get_state_index(self.states[i])
            action = self.actions[i]
            reward = discounted_rewards[i]
            
            # 计算当前策略下的动作概率
            probs = self.get_action_probs(self.states[i])
            
            # 计算梯度
            dsoftmax = probs.copy()
            dsoftmax[action] -= 1
            
            # 更新权重
            self.weights[state_idx] -= self.learning_rate * reward * dsoftmax
        
        # 清空经验池
        self.states = []
        self.actions = []
        self.rewards = []
        
        # 降低探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def train(self, episodes=1000):
        """训练智能体"""
        rewards_history = []
        
        for episode in range(episodes):
            # 每个episode开始时更新动态障碍物
            self.env.update_dynamic_obstacles()
            
            # 如果使用时间敏感奖励函数，需要重置episode_steps
            if self.reward_version == 'v5':
                self.env.episode_steps = 0
            
            # 初始化状态
            state = self.env.start
            total_reward = 0
            steps = 0
            
            while state != self.env.goal:
                action = self.get_action(state)
                next_state = self.get_next_state(state, action)
                
                # 使用新的奖励函数计算奖励
                reward = self.calculate_reward(state, action, next_state)
                
                # 存储经验
                self.store_experience(state, action, reward)
                
                # 更新状态
                state = next_state
                total_reward += reward
                steps += 1
                
                # 如果使用时间敏感奖励函数，需要更新episode_steps
                if self.reward_version == 'v5':
                    self.env.episode_steps += 1
                
                if steps > 1000:  # 防止无限循环
                    break
            
            # 更新策略
            self.update_policy()
            rewards_history.append(total_reward)
            
            # 每200个episode输出一次当前最优路径（使用贪婪策略找到的路径）
            # 修改为从第200个episode开始显示图像
            if episode % 200 == 0 and episode >= 200:
                print(f"Policy Gradient Episode {episode}, Reward: {total_reward}")
                # 可视化当前最优路径（使用贪婪策略找到的路径）
                path = self.find_path()
                # 验证路径的合法性
                valid_path = self.validate_path(path)
                if valid_path:
                    self.env.visualize(path, f'Policy Gradient Path - Episode {episode} - Reward {self.reward_version}')
        
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
        """找到从起点到终点的路径（使用确定性策略）"""
        path = []
        state = self.env.start
        visited_states = set()
        
        # 在查找路径时需要考虑当前动态障碍物的位置
        max_steps = 1000
        steps = 0
        
        while state != self.env.goal and steps < max_steps:
            # 使用确定性策略（选择概率最高的动作）
            state_idx = self.get_state_index(state)
            logits = self.weights[state_idx]
            action = np.argmax(logits)
            
            next_state = self.get_next_state(state, action)
            
            # 检查是否形成循环
            if next_state in visited_states:
                # 尝试其他动作
                best_action = action
                best_logit = logits[best_action]
                found_better = False
                for a in range(4):
                    test_logit = logits[a]
                    test_next = self.get_next_state(state, a)
                    if test_next not in visited_states and test_logit > best_logit:
                        best_action = a
                        best_logit = test_logit
                        next_state = test_next
                        found_better = True
                
                # 如果所有动作都会导致循环，就跳出
                if not found_better:
                    break
            
            path.append(state)
            visited_states.add(state)
            state = next_state
            steps += 1
            
        # 只有当最后一步是终点时才添加终点
        if state == self.env.goal:
            path.append(self.env.goal)
        return path

# 测试策略梯度算法
if __name__ == "__main__":
    from environment import GridEnvironment
    import matplotlib.pyplot as plt
    
    # 设置随机种子以确保结果可复现
    random.seed(42)
    np.random.seed(42)
    
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
    
    # 创建策略梯度智能体
    policy_agent = PolicyGradientAgent(env)
    
    # 训练智能体
    print("Training Policy Gradient agent...")
    policy_rewards = policy_agent.train(episodes=1000)
    
    # 找到路径（在查找路径前更新动态障碍物位置）
    env.update_dynamic_obstacles()  # 确保我们看到的是当前的动态障碍物位置
    policy_path = policy_agent.find_path()
    
    # 验证路径合法性
    policy_valid = policy_agent.validate_path(policy_path)
    
    print(f"Policy Gradient path valid: {policy_valid}")
    
    # 可视化结果
    if policy_path and policy_valid:
        env.visualize(policy_path, 'Policy Gradient Path Planning')
    elif policy_path:
        print("Warning: Policy Gradient generated an invalid path")
        env.visualize(policy_path, 'Policy Gradient Path Planning (Invalid)')
    else:
        env.visualize([], 'Policy Gradient Path Planning (No Path Found)')
    
    # 绘制奖励曲线
    plt.figure(figsize=(10, 5))
    
    # 计算滑动平均以更好地显示趋势
    window_size = 50
    policy_rewards_smooth = np.convolve(policy_rewards, np.ones(window_size)/window_size, mode='valid')
    
    episodes_range = np.arange(len(policy_rewards_smooth))
    plt.plot(episodes_range, policy_rewards_smooth, label='Policy Gradient', color='orange', linewidth=2)
    plt.title('Policy Gradient Cumulative Reward over Episodes', fontsize=14, fontweight='bold')
    plt.xlabel('Episode')
    plt.ylabel('Reward (Smoothed)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()
    
    # 输出统计信息
    print("\n=== Policy Gradient Performance ===")
    print(f"Policy Gradient - Path length: {len(policy_path)}")
    print(f"Policy Gradient - Final reward: {policy_rewards[-1]}")
    print(f"Policy Gradient - Average reward (last 100 episodes): {np.mean(policy_rewards[-100:]):.2f}")