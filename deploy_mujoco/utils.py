import mujoco
import numpy as np


class TrajectoryDrawer:
    def __init__(self, max_segments=500, min_distance=0.01):
        """
        轨迹绘制器的初始化。

        参数：
        - max_segments: 最大显示的轨迹线段数
        - min_distance: 相邻轨迹点之间的最小距离
        """
        self.max_segments = max_segments
        self.min_distance = min_distance
        self.positions = []  # 存储有效的轨迹点
        self.current_point = None  # 当前点
        self.accumulated_distance = 0.0  # 累积距离

    def add_point(self, point):
        """
        添加新的轨迹点。

        参数：
        - point: 新的位置点
        """
        point = np.array(point)

        # 如果是第一个点
        if not self.positions:
            self.positions.append(point.copy())
            self.current_point = point.copy()
            return

        # 如果已经有current_point，计算与current_point的距离
        if self.current_point is not None:
            distance = np.linalg.norm(point - self.current_point)
            self.accumulated_distance += distance
            self.current_point = point.copy()

            # 如果累积距离超过最小距离，将current_point添加为有效点
            if self.accumulated_distance >= self.min_distance:
                self.positions.append(self.current_point.copy())
                self.accumulated_distance = 0.0  # 重置累积距离

                # 如果超过最大段数，移除最早的点
                if len(self.positions) > self.max_segments + 1:
                    self.positions.pop(0)

    def clean_dense_points(self, target_distance=None):
        """
        清理轨迹中过于密集的点。

        参数：
        - target_distance: 目标距离，如果不指定则使用 min_distance
        """
        if target_distance is None:
            target_distance = self.min_distance

        if len(self.positions) < 2:
            return

        i = 0
        while i < len(self.positions) - 1:
            j = i + 1
            while j < len(self.positions):
                distance = np.linalg.norm(
                    np.array(self.positions[j]) - np.array(self.positions[i])
                )
                if distance < target_distance:
                    self.positions.pop(j)
                else:
                    i = j
                    break
            i += 1

    def draw_trajectory(
        self,
        viewer,
        color=[0, 1, 0, 1],
        width=0.002,
        fade=False,
        dynamic_color=[0, 1, 0, 1],
        dynamic_width=0.002,
    ):
        """
        绘制完整轨迹。

        参数：
        - viewer: Mujoco 查看器对象
        - color: 基础颜色 [r,g,b,a]
        - width: 线条宽度
        - fade: 是否启用渐变效果
        - dynamic_color: 动态轨迹段的颜色
        - dynamic_width: 动态轨迹段的宽度
        """
        if len(self.positions) < 1:
            return

        # 绘制固定轨迹
        for i in range(len(self.positions) - 1):
            if fade:
                alpha = (i / (len(self.positions) - 1)) * color[3]
            else:
                alpha = color[3]

            current_color = [color[0], color[1], color[2], alpha]

            mujoco.mjv_connector(
                viewer.user_scn.geoms[viewer.user_scn.ngeom],
                type=mujoco.mjtGeom.mjGEOM_LINE,
                width=width,
                from_=self.positions[i],
                to=self.positions[i + 1],
            )
            viewer.user_scn.geoms[viewer.user_scn.ngeom].rgba = current_color
            viewer.user_scn.ngeom += 1

        # 绘制动态轨迹段（从最后一个固定点到当前点）
        if self.current_point is not None and len(self.positions) > 0:
            mujoco.mjv_connector(
                viewer.user_scn.geoms[viewer.user_scn.ngeom],
                type=mujoco.mjtGeom.mjGEOM_LINE,
                width=dynamic_width,
                from_=self.positions[-1],
                to=self.current_point,
            )
            viewer.user_scn.geoms[viewer.user_scn.ngeom].rgba = dynamic_color
            viewer.user_scn.ngeom += 1
