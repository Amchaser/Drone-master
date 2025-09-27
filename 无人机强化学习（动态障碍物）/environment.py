import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle

class GridEnvironment:
    def __init__(self, size=20):
        self.size = size
        self.grid = np.zeros((size, size))
        self.start = (0, 0)  # 左下角
        self.goal = (size-1, size-1)  # 右上角
        self.obstacles = []
        self.dynamic_obstacles = []  # 动态障碍物列表
        self.timestep = 0  # 时间步
        
    def add_obstacle(self, x, y, width, height):
        """添加静态障碍物"""
        for i in range(x, x + width):
            for j in range(y, y + height):
                if 0 <= i < self.size and 0 <= j < self.size:
                    self.grid[i, j] = 1
        self.obstacles.append((x, y, width, height))
    
    def add_dynamic_obstacles(self, num_obstacles=2, obstacle_size=1):
        """添加固定轨迹往复运动的动态障碍物，初始位置固定"""
        for i in range(num_obstacles):
            # 使用传入的障碍物大小
            size = obstacle_size
            
            # 固定移动方向和轨迹
            if i == 0:
                # 第一个障碍物水平移动
                row = 4  # 固定在第4行
                # 限制移动范围，确保不会阻挡起点到终点的主要路径
                trajectory = [(row, col) for col in range(4, self.size - 4)]
                start_index = 0  # 固定初始位置在最左端
                direction = 1    # 初始移动方向向右
            else:
                # 第二个障碍物垂直移动
                col = 16  # 固定在第16列
                # 限制移动范围，确保不会阻挡起点到终点的主要路径
                trajectory = [(row, col) for row in range(4, self.size - 4)]
                start_index = 0  # 固定初始位置在最上端
                direction = 1    # 初始移动方向向下
            
            dynamic_obstacle = {
                'trajectory': trajectory,
                'current_index': start_index,
                'size': size,
                'direction': direction  # 1表示正向移动，-1表示反向移动
            }
            self.dynamic_obstacles.append(dynamic_obstacle)
    
    def update_dynamic_obstacles(self):
        """更新动态障碍物位置"""
        self.timestep += 1
        for obstacle in self.dynamic_obstacles:
            # 移动到下一个位置
            obstacle['current_index'] += obstacle['direction']
            
            # 如果到达轨迹的末端，改变方向
            if obstacle['current_index'] >= len(obstacle['trajectory']) - 1:
                obstacle['current_index'] = len(obstacle['trajectory']) - 1
                obstacle['direction'] = -1
            elif obstacle['current_index'] <= 0:
                obstacle['current_index'] = 0
                obstacle['direction'] = 1
            # 确保索引在有效范围内
            obstacle['current_index'] = max(0, min(len(obstacle['trajectory']) - 1, obstacle['current_index']))

    def get_dynamic_obstacle_positions(self):
        """获取当前动态障碍物占据的位置"""
        positions = []
        for obstacle in self.dynamic_obstacles:
            # 获取当前轨迹点
            center_x, center_y = obstacle['trajectory'][obstacle['current_index']]
            size = int(obstacle.get('size', 1))  # 确保大小为整数
            
            # 计算障碍物占据的所有网格位置（以中心点为中心）
            # 对于奇数大小（如1x1, 3x3），中心是网格点
            # 对于偶数大小（如2x2, 4x4），中心在网格线交点
            if size % 2 == 1:  # 奇数大小
                half_size = size // 2
                for i in range(center_x - half_size, center_x + half_size + 1):
                    for j in range(center_y - half_size, center_y + half_size + 1):
                        if 0 <= i < self.size and 0 <= j < self.size:
                            positions.append((i, j))
            else:  # 偶数大小
                half_size = size // 2
                # 对于偶数大小，我们需要从中心点向左上角扩展
                for i in range(center_x - half_size + 1, center_x + half_size + 1):
                    for j in range(center_y - half_size + 1, center_y + half_size + 1):
                        if 0 <= i < self.size and 0 <= j < self.size:
                            positions.append((i, j))
        return positions
    
    def is_valid_position(self, x, y):
        """检查位置是否有效（不在障碍物中且在边界内）"""
        if not (0 <= x < self.size and 0 <= y < self.size):
            return False
        # 检查是否在静态障碍物中
        if self.grid[x, y] == 1:
            return False
        # 检查是否在动态障碍物中
        dynamic_positions = self.get_dynamic_obstacle_positions()
        if (x, y) in dynamic_positions:
            return False
        return True
    
    def visualize(self, path=None, title='Grid Environment'):
        """可视化环境"""
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # 创建一个临时网格用于显示，包含动态障碍物
        display_grid = self.grid.copy()
        dynamic_positions = self.get_dynamic_obstacle_positions()
        for x, y in dynamic_positions:
            if 0 <= x < self.size and 0 <= y < self.size:
                display_grid[x, y] = 0.5  # 动态障碍物用不同的值表示
        
        ax.imshow(display_grid, cmap='binary', origin='lower')
        
        # 标记起点和终点
        ax.plot(self.start[1], self.start[0], 'go', markersize=10, label='Start')
        ax.plot(self.goal[1], self.goal[0], 'ro', markersize=10, label='Goal')
        
        # 绘制动态障碍物轨迹
        for obstacle in self.dynamic_obstacles:
            trajectory_x = [pos[1] for pos in obstacle['trajectory']]
            trajectory_y = [pos[0] for pos in obstacle['trajectory']]
            ax.plot(trajectory_x, trajectory_y, 'c--', alpha=0.5, linewidth=1)
        
        # 标记路径
        if path:
            path_x = [p[1] for p in path]
            path_y = [p[0] for p in path]
            ax.plot(path_x, path_y, 'b-', linewidth=2, label='Path')
        
        # 标记动态障碍物中心
        for obstacle in self.dynamic_obstacles:
            center_x, center_y = obstacle['trajectory'][obstacle['current_index']]
            ax.plot(center_y, center_x, 'rx', markersize=8)
        
        ax.set_xlim(0, self.size)
        ax.set_ylim(0, self.size)
        ax.set_xticks(range(self.size))
        ax.set_yticks(range(self.size))
        ax.grid(True)
        ax.legend()
        plt.title(title)
        plt.show()
        
    def visualize_dynamic_path(self, path, title='Dynamic Path Visualization', interval=500):
        """动态可视化路径和移动的障碍物"""
        if not path:
            print("路径为空，无法进行动态可视化")
            return
            
        # 创建图形和轴
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # 用于存储路径点的坐标
        path_x = [p[1] for p in path]
        path_y = [p[0] for p in path]
        
        # 获取动态障碍物轨迹信息
        dynamic_obstacles_info = []
        for obstacle in self.dynamic_obstacles:
            dynamic_obstacles_info.append({
                'trajectory': obstacle['trajectory'],
                'size': obstacle['size']
            })
        
        # 计算总的动画帧数（路径点数 + 一些额外的帧用于循环展示）
        total_frames = len(path) * 2  # 每个步骤显示2帧，使动画更流畅
        
        def animate(frame):
            ax.clear()
            
            # 显示静态网格
            display_grid = np.zeros((self.size, self.size))
            for i in range(self.size):
                for j in range(self.size):
                    display_grid[i, j] = self.grid[i, j]
            
            ax.imshow(display_grid, cmap='binary', origin='lower')
            
            # 标记起点和终点
            ax.plot(self.start[1], self.start[0], 'go', markersize=12, label='Start')
            ax.plot(self.goal[1], self.goal[0], 'r*', markersize=15, label='Goal')
            
            # 绘制静态障碍物
            for (row, col, h, w) in self.obstacles:
                rect = Rectangle((col - 0.5, row - 0.5), w, h, linewidth=1, 
                                edgecolor='gray', facecolor='gray', alpha=0.7)
                ax.add_patch(rect)
            
            # 计算当前路径点和动态障碍物位置
            path_frame = min(frame // 2, len(path) - 1)  # 每个路径点显示2帧
            
            # 更新动态障碍物位置
            temp_env = GridEnvironment(self.size)
            temp_env.grid = self.grid.copy()
            temp_env.obstacles = self.obstacles
            temp_env.dynamic_obstacles = []
            
            # 复制动态障碍物并更新它们的位置
            for i, obstacle_info in enumerate(dynamic_obstacles_info):
                # 计算当前帧的动态障碍物位置
                trajectory = obstacle_info['trajectory']
                trajectory_length = len(trajectory)
                
                # 使用正弦函数创建更自然的来回移动效果
                # 添加偏移量使不同障碍物有不同相位
                offset = i * (trajectory_length // 3)
                progress = (frame + offset) % (2 * trajectory_length - 2)
                if progress >= trajectory_length:
                    current_index = 2 * trajectory_length - 2 - progress
                else:
                    current_index = progress
                    
                current_index = max(0, min(trajectory_length - 1, current_index))
                
                # 创建临时动态障碍物对象
                temp_obstacle = {
                    'trajectory': trajectory,
                    'current_index': current_index,
                    'size': obstacle_info['size'],
                    'direction': 1
                }
                temp_env.dynamic_obstacles.append(temp_obstacle)
            
            # 获取当前动态障碍物位置并绘制
            dynamic_positions = temp_env.get_dynamic_obstacle_positions()
            # 绘制动态障碍物（根据实际大小绘制）
            for obstacle in temp_env.dynamic_obstacles:
                center_y, center_x = obstacle['trajectory'][obstacle['current_index']]  # 注意坐标顺序
                size = int(obstacle.get('size', 1))  # 确保大小为整数
                half_size = size / 2.0
                
                # 根据障碍物大小计算绘制位置
                if size % 2 == 1:  # 奇数大小
                    # 奇数大小的障碍物以网格点为中心
                    rect_x = center_x - half_size
                    rect_y = center_y - half_size
                else:  # 偶数大小
                    # 偶数大小的障碍物以网格线交点为中心
                    rect_x = center_x - half_size + 0.5
                    rect_y = center_y - half_size + 0.5
                
                # 绘制障碍物矩形
                rect = Rectangle((rect_x, rect_y), 
                               size, size, linewidth=1,
                               edgecolor='red', facecolor='red', alpha=0.7)
                ax.add_patch(rect)
            
            # 绘制路径（已经走过的部分用一种颜色，未来的部分用另一种颜色）
            # 已经走过的路径
            past_path_x = path_x[:path_frame+1]
            past_path_y = path_y[:path_frame+1]
            ax.plot(past_path_x, past_path_y, 'b-', linewidth=3, marker='o', markersize=6, label='Past Path')
            
            # 当前位置（无人机位置）
            current_x, current_y = path_x[path_frame], path_y[path_frame]
            ax.plot(current_y, current_x, 'bo', markersize=12, markeredgecolor='yellow', markeredgewidth=2, label='UAV')
            
            # 未来路径（如果有的话）
            if path_frame + 1 < len(path):
                future_path_x = path_x[path_frame:]
                future_path_y = path_y[path_frame:]
                ax.plot(future_path_x, future_path_y, 'b--', linewidth=2, alpha=0.5, label='Future Path')
            
            ax.set_xlim(-0.5, self.size - 0.5)
            ax.set_ylim(-0.5, self.size - 0.5)
            ax.set_xticks(range(self.size))
            ax.set_yticks(range(self.size))
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right')
            
            # 显示进度信息
            progress_info = f'{title} - Step {path_frame + 1}/{len(path)}'
            ax.set_title(progress_info, fontsize=14)
        
        # 创建动画
        anim = FuncAnimation(fig, animate, frames=total_frames, interval=interval, repeat=True, blit=False)
        
        plt.tight_layout()
        plt.show()
        
        return anim