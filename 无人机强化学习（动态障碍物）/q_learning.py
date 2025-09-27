import numpy as np
import random

class QLearningAgent:
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
    
    def update_q_value(self, state, action, reward, next_state):
        """更新Q值"""
        current_q = self.q_table[state[0], state[1], action]
        max_next_q = np.max(self.q_table[next_state[0], next_state[1]])
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
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
            total_reward = 0
            steps = 0
            
            while state != self.env.goal:
                action = self.get_action(state)
                next_state = self.get_next_state(state, action)
                
                # 使用新的奖励函数计算奖励
                reward = self.calculate_reward(state, action, next_state)
                
                self.update_q_value(state, action, reward, next_state)
                state = next_state
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
                    self.env.visualize(path, f'Q-Learning Path - Episode {episode} - Reward {self.reward_version}')
        
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
        
        # 在查找路径时需要考虑当前动态障碍物的位置
        max_steps = self.env.size * self.env.size  # 设置最大步数限制
        steps = 0
        
        while state != self.env.goal and steps < max_steps:
            action = np.argmax(self.q_table[state[0], state[1]])
            next_state = self.get_next_state(state, action)
            
            # 检查是否形成循环
            if next_state in visited_states:
                # 尝试其他动作
                best_action = action
                best_next_state = next_state
                found_better = False
                
                # 按Q值排序动作，尝试Q值较高的其他动作
                action_q_values = self.q_table[state[0], state[1]]
                sorted_actions = np.argsort(action_q_values)[::-1]  # 降序排列
                
                for a in sorted_actions:
                    test_next = self.get_next_state(state, a)
                    if test_next not in visited_states and test_next != state:
                        best_action = a
                        best_next_state = test_next
                        found_better = True
                        break
                
                # 如果所有动作都会导致循环或碰撞，尝试随机动作
                if not found_better:
                    for _ in range(10):  # 最多尝试10次随机动作
                        random_action = random.randint(0, 3)
                        test_next = self.get_next_state(state, random_action)
                        if test_next not in visited_states and test_next != state:
                            best_action = random_action
                            best_next_state = test_next
                            found_better = True
                            break
                
                # 如果还是找不到合适的动作，则跳出循环
                if not found_better:
                    break
                    
                action = best_action
                next_state = best_next_state
            
            path.append(state)
            visited_states.add(state)
            state = next_state
            steps += 1
            
            # 防止无限循环的额外检查
            if steps > max_steps:
                break
        
        # 只有当最后一步是终点时才添加终点
        if state == self.env.goal:
            path.append(self.env.goal)
        return path
