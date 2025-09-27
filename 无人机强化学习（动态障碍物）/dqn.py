import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque

class DQN(nn.Module):
    """深度Q网络模型"""
    def __init__(self, input_size=4, hidden_size=128, output_size=4):  # 恢复网络大小
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size//2)
        self.fc4 = nn.Linear(hidden_size//2, output_size)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class DQNAgent:
    def __init__(self, env, learning_rate=0.001, discount_factor=0.95, epsilon=0.9, 
                 epsilon_decay=0.995, epsilon_min=0.01, reward_version='v1'):
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon  # 提高初始探索率，促进更多探索
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.epsilon_max = epsilon  # 添加epsilon_max属性
        
        # 设备选择
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 神经网络
        self.q_network = DQN().to(self.device)
        self.target_network = DQN().to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # 初始化目标网络
        self.update_target_network()
        
        # 经验回放
        self.memory = deque(maxlen=10000)
        self.batch_size = 32
        
        # 增加目标网络更新频率，提高训练稳定性
        self.target_update_freq = 50
        self.step_count = 0
        
        # 导入奖励函数模块
        from reward_functions import get_reward_function
        self.reward_function = get_reward_function(reward_version)
        self.reward_version = reward_version
        
        # 启用双DQN以减少Q值过高估计问题
        self.use_double_dqn = True
        
    def get_state(self, position):
        """将位置转换为状态向量"""
        # 状态包括当前位置和目标位置的归一化坐标
        if isinstance(position, tuple):
            x, y = position
        else:
            x, y = position[0], position[1]
            
        state = torch.FloatTensor([
            x / self.env.size,
            y / self.env.size,
            self.env.goal[0] / self.env.size,
            self.env.goal[1] / self.env.size
        ]).to(self.device)
        return state
    
    def get_action(self, state):
        """选择动作"""
        if random.random() < self.epsilon:
            return random.randint(0, 3)  # 随机动作
        else:
            with torch.no_grad():
                # 确保state是torch张量
                if not isinstance(state, torch.Tensor):
                    state = self.get_state(state)
                # 确保是二维张量(batch_size, feature_dim)
                if state.dim() == 1:
                    state = state.unsqueeze(0)
                q_values = self.q_network(state)
                return q_values.argmax(dim=1).item()
    
    def remember(self, state, action, reward, next_state, done):
        """存储经验"""
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self):
        """经验回放"""
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states = torch.stack([e[0] for e in batch])
        actions = torch.LongTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.stack([e[3] for e in batch])
        dones = torch.BoolTensor([e[4] for e in batch]).to(self.device)
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = torch.zeros(self.batch_size, device=self.device)
        
        with torch.no_grad():
            if self.use_double_dqn:
                # 双DQN: 使用主网络选择动作，目标网络评估动作值
                next_actions = self.q_network(next_states).max(1)[1]
                next_q_values[~dones] = self.target_network(next_states[~dones]).gather(1, next_actions[~dones].unsqueeze(1)).squeeze()
            else:
                # 标准DQN: 使用目标网络选择和评估动作
                next_q_values[~dones] = self.target_network(next_states[~dones]).max(1)[0]
        
        target_q_values = rewards + (self.discount_factor * next_q_values)
        
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        # 添加梯度裁剪以提高训练稳定性
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # 更新epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def update_target_network(self):
        """更新目标网络"""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def get_next_state(self, state, action):
        """获取下一个状态"""
        # 将状态转换为位置
        x, y = None, None
        
        if isinstance(state, tuple) and len(state) >= 2:
            x, y = state[0], state[1]
        elif isinstance(state, torch.Tensor):
            if state.dim() == 0:
                # 标量张量
                x, y = int(state.item()), int(state.item())
            elif state.dim() == 1 and len(state) >= 2:
                x = int(state[0].item() * self.env.size)
                y = int(state[1].item() * self.env.size)
            else:
                # 其他情况默认为0
                x, y = 0, 0
        elif isinstance(state, (list, np.ndarray)) and len(state) >= 2:
            x = int(state[0] * self.env.size)
            y = int(state[1] * self.env.size)
        elif hasattr(state, '__getitem__') and len(state) >= 2:
            x = int(state[0] * self.env.size) if isinstance(state[0], (int, float)) else 0
            y = int(state[1] * self.env.size) if isinstance(state[1], (int, float)) else 0
        else:
            # 如果以上都不匹配，默认使用起始位置
            x, y = self.env.start[0], self.env.start[1]
            
        # 确保x和y是整数
        if not isinstance(x, int):
            x = int(x) if isinstance(x, (float, np.floating)) else 0
        if not isinstance(y, int):
            y = int(y) if isinstance(y, (float, np.floating)) else 0
        
        # 确保x和y在合法范围内
        x = max(0, min(self.env.size - 1, x))
        y = max(0, min(self.env.size - 1, y))
        
        if action == 0:  # 上
            next_x, next_y = x - 1, y
        elif action == 1:  # 右
            next_x, next_y = x, y + 1
        elif action == 2:  # 下
            next_x, next_y = x + 1, y
        elif action == 3:  # 左
            next_x, next_y = x, y - 1
        else:
            next_x, next_y = x, y  # 无效动作，保持原位
            
        # 确保新位置在网格范围内
        next_x = max(0, min(self.env.size - 1, next_x))
        next_y = max(0, min(self.env.size - 1, next_y))
        
        return (next_x, next_y)
    
    def calculate_reward(self, state, action, next_state):
        """
        计算奖励值
        
        Args:
            state: 当前状态 (坐标元组)
            action: 执行的动作
            next_state: 下一状态 (坐标元组)
            
        Returns:
            reward: 奖励值
        """
        # 确保state是坐标元组
        if isinstance(state, torch.Tensor):
            if state.dim() == 1 and len(state) >= 2:
                x = int(state[0].item() * self.env.size)
                y = int(state[1].item() * self.env.size)
                state = (x, y)
            else:
                state = self.env.start
                
        # 确保next_state是坐标元组
        if isinstance(next_state, torch.Tensor):
            if next_state.dim() == 1 and len(next_state) >= 2:
                x = int(next_state[0].item() * self.env.size)
                y = int(next_state[1].item() * self.env.size)
                next_state = (x, y)
            else:
                next_state = self.env.start
        
        goal_reached = (next_state == self.env.goal)
        collision = (state == next_state and next_state != self.env.goal)
        
        return self.reward_function(
            self.env, state, action, next_state, 
            goal_reached=goal_reached, 
            collision=collision
        )
    
    def select_action(self, state_tensor, episode, total_episodes):
        """
        选择动作
        
        Args:
            state_tensor: 当前状态的张量表示
            episode: 当前episode数
            total_episodes: 总episode数
            
        Returns:
            action: 选择的动作
        """
        # 确保state_tensor有正确的维度
        if state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0)
        
        # epsilon-greedy策略
        # 动态调整epsilon，随着训练进行逐渐减少探索
        progress = episode / total_episodes
        current_epsilon = self.epsilon_min + (self.epsilon_max - self.epsilon_min) * np.exp(-5 * progress)
        
        if random.random() < current_epsilon:
            # 随机选择动作
            return random.randint(0, 3)
        else:
            # 使用Q网络选择最佳动作
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
                return q_values.argmax().item()

    def train(self, episodes=1000):
        """训练智能体"""
        rewards_history = []
        
        # 初始化目标网络
        self.update_target_network()
        
        # 添加经验回放计数器
        experience_counter = 0
        
        for episode in range(episodes):
            # 每个episode开始时更新动态障碍物
            self.env.update_dynamic_obstacles()
            
            # 如果使用时间敏感奖励函数，需要重置episode_steps
            if self.reward_version == 'v5':
                self.env.episode_steps = 0
            
            # 初始化状态
            state = self.env.start
            state_tensor = self.get_state(state)
            total_reward = 0
            steps = 0
            
            # 限制episode最大步数以防止无限循环
            max_steps_per_episode = 500
            
            while state != self.env.goal and steps < max_steps_per_episode:
                # 选择动作
                action = self.select_action(state_tensor, episode, episodes)
                
                # 执行动作
                next_state = self.get_next_state(state, action)
                
                # 检查是否与障碍物碰撞
                collision = not self.env.is_valid_position(next_state[0], next_state[1]) and next_state != self.env.goal
                
                # 检查是否到达目标
                goal_reached = (next_state == self.env.goal)
                
                # 计算奖励
                reward = self.reward_function(
                    self.env, state, action, next_state,
                    goal_reached=goal_reached,
                    collision=collision
                )
                
                # 获取下一个状态的张量表示
                next_state_tensor = self.get_state(next_state)
                
                # 存储经验
                self.memory.append((
                    state_tensor.squeeze(0),  # 移除批次维度
                    action,
                    reward,
                    next_state_tensor.squeeze(0),  # 移除批次维度
                    goal_reached or steps >= max_steps_per_episode - 1
                ))
                
                # 增加经验计数器
                experience_counter += 1
                
                # 经验回放 - 每4步进行一次
                if len(self.memory) >= self.batch_size and experience_counter % 4 == 0:
                    self.replay()
                
                # 更新状态和累积奖励
                state = next_state
                state_tensor = next_state_tensor
                total_reward += reward
                steps += 1
                
                # 更新目标网络
                if self.step_count % self.target_update_freq == 0:
                    self.update_target_network()
                
                self.step_count += 1
            
            rewards_history.append(total_reward)
            
            # 修复可视化输出逻辑，确保正确显示训练进度
            visualization_frequency = max(1, episodes // 10)  # 可视化频率为总episode数的1/10
            if episodes >= 100 and episode % visualization_frequency == 0:
                print(f"DQN Episode {episode}, Reward: {total_reward}")
            elif episodes < 100:  # 如果训练次数少于100，每10次显示一次
                if episode % 10 == 0:
                    print(f"DQN Episode {episode}, Reward: {total_reward}")
        
        return rewards_history
    
    def validate_path(self, path):
        """验证路径是否合法（不穿越障碍物）"""
        if not path:
            return False
        
        # 检查起点和终点
        if path[0] != self.env.start:
            return False
        if path[-1] != self.env.goal:
            return False
        
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
                
            # 检查移动是否符合规则（只能上下左右移动一格）
            x1, y1 = state
            x2, y2 = next_state
            dx, dy = abs(x2 - x1), abs(y2 - y1)
            
            # 必须是相邻的格子（曼哈顿距离为1）
            if dx + dy != 1:
                return False
        
        return True
    
    def find_path(self):
        """找到从起点到终点的路径"""
        path = []
        state = self.env.start
        visited_states = set()
        
        # 在查找路径时需要考虑当前动态障碍物的位置
        max_steps = 1000  # 增加最大步数限制
        steps = 0
        
        while state != self.env.goal and steps < max_steps:
            state_tensor = self.get_state(state)
            with torch.no_grad():
                # 确保state_tensor是正确的维度
                if state_tensor.dim() == 1:
                    state_tensor = state_tensor.unsqueeze(0)
                q_values = self.q_network(state_tensor)
                action = q_values.argmax().item()
            
            next_state = self.get_next_state(state, action)
            
            # 检查下一个状态是否有效（不与障碍物碰撞）
            if not self.env.is_valid_position(next_state[0], next_state[1]) and next_state != self.env.goal:
                # 如果下一个状态是障碍物，尝试其他动作
                found_valid_action = False
                for a in range(4):
                    test_next = self.get_next_state(state, a)
                    if (test_next not in visited_states and 
                        (self.env.is_valid_position(test_next[0], test_next[1]) or test_next == self.env.goal)):
                        next_state = test_next
                        found_valid_action = True
                        break
                
                # 如果找不到有效动作，跳出循环
                if not found_valid_action:
                    break
            
            # 检查是否形成循环
            if next_state in visited_states:
                # 尝试其他动作
                found_better = False
                for a in range(4):
                    test_next = self.get_next_state(state, a)
                    if (test_next not in visited_states and 
                        (self.env.is_valid_position(test_next[0], test_next[1]) or test_next == self.env.goal)):
                        next_state = test_next
                        found_better = True
                        break
                
                # 如果所有动作都会导致循环，尝试随机动作
                if not found_better:
                    for _ in range(10):  # 尝试最多10次随机动作
                        random_action = random.randint(0, 3)
                        test_next = self.get_next_state(state, random_action)
                        if (test_next not in visited_states and test_next != state and
                            (self.env.is_valid_position(test_next[0], test_next[1]) or test_next == self.env.goal)):
                            next_state = test_next
                            found_better = True
                            break
                
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

# 测试DQN算法
if __name__ == "__main__":
    from environment import GridEnvironment
    import matplotlib.pyplot as plt
    
    # 创建环境
    env = GridEnvironment(size=20)

    # 添加静态障碍物（至少10个，分散布置，减小部分障碍物面积）
    obstacles = [
        (3, 3, 3, 3),  # 左上区域 (减小面积 4x4 -> 3x3)
        (15, 3, 3, 3),  # 右上区域 (减小面积 4x4 -> 3x3)
        (3, 15, 3, 3),  # 左下区域 (减小面积 4x4 -> 3x3)
        (15, 15, 3, 3),  # 右下区域 (减小面积 4x4 -> 3x3)
        (8, 8, 2, 2),  # 中心偏左 (减小面积 3x3 -> 2x2)
        (12, 8, 2, 2),  # 中心偏右 (减小面积 3x3 -> 2x2)
        (8, 12, 2, 2),  # 中心偏下 (减小面积 3x3 -> 2x2)
        (12, 12, 2, 2),  # 中心偏上 (减小面积 3x3 -> 2x2)
        (5, 10, 2, 1),  # 左侧中部 (减小面积 2x2 -> 2x1)
        (15, 10, 2, 1)  # 右侧中部 (减小面积 2x2 -> 2x1)
    ]

    # 在边界添加额外的静态障碍物，但避开起点(0,0)和终点(19,19)附近区域
    # 上边界障碍物 (避开起点附近区域)
    obstacles.append((0, 4, 1, 2))   # 下左
    #obstacles.append((0, 14, 1, 2))  # 下右

    # 下边界障碍物 (避开终点附近区域)
    obstacles.append((19, 2, 1, 2))  # 上左
    #obstacles.append((19, 14, 1, 2)) # 上右

    # 左边界障碍物 (避开起点附近区域)
    #obstacles.append((2, 0, 2, 1))   # 左边界下
    obstacles.append((14, 0, 2, 1))  # 左边界上

    # 右边界障碍物 (避开终点附近区域)
    obstacles.append((2, 19, 2, 1))  # 右边界下侧
    #obstacles.append((14, 19, 2, 1)) # 右边界上侧

    # 添加障碍物到环境中
    for obs in obstacles:
        env.add_obstacle(*obs)
    # 添加固定轨迹往复运动的动态障碍物
    env.add_dynamic_obstacles(2)
    
    # 创建DQN智能体
    dqn_agent = DQNAgent(env)
    
    # 训练智能体
    print("Training DQN agent...")
    dqn_rewards = dqn_agent.train(episodes=400)
    
    # 找到路径（在查找路径前更新动态障碍物位置）
    env.update_dynamic_obstacles()  # 确保我们看到的是当前的动态障碍物位置
    dqn_path = dqn_agent.find_path()
    
    # 验证路径合法性
    dqn_valid = dqn_agent.validate_path(dqn_path)
    
    print(f"DQN path valid: {dqn_valid}")
    
    # 可视化结果
    if dqn_path and dqn_valid:
        env.visualize(dqn_path, 'DQN Path Planning')
    elif dqn_path:
        print("Warning: DQN generated an invalid path")
        env.visualize(dqn_path, 'DQN Path Planning (Invalid)')
    else:
        env.visualize([], 'DQN Path Planning (No Path Found)')
    
    # 绘制奖励曲线
    plt.figure(figsize=(10, 5))
    
    # 计算滑动平均以更好地显示趋势
    window_size = 50
    dqn_rewards_smooth = np.convolve(dqn_rewards, np.ones(window_size)/window_size, mode='valid')
    
    episodes_range = np.arange(len(dqn_rewards_smooth))
    plt.plot(episodes_range, dqn_rewards_smooth, label='DQN', color='green', linewidth=2)
    plt.title('DQN Cumulative Reward over Episodes', fontsize=14, fontweight='bold')
    plt.xlabel('Episode')
    plt.ylabel('Reward (Smoothed)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()
    
    # 输出统计信息
    print("\n=== DQN Performance ===")
    print(f"DQN - Path length: {len(dqn_path)}")
    print(f"DQN - Final reward: {dqn_rewards[-1]}")
    print(f"DQN - Average reward (last 100 episodes): {np.mean(dqn_rewards[-100:]):.2f}")