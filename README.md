# mujoco_unitree_rl
combine `rl_sar` and `unitree_mujoco` repo, you need to change the `LEGGED_GYM_ROOT_DIR` value to your path in file `rl_sdk.py`.

example code
```bash
cd deploy_mujoco
python main.py a1_gym.yaml
```

键盘操作示例
```bash
enter: 开始/结束模拟
数字 1: 站起
数字 4: 趴下
P : 开启RL
W/A/S/D/J/K/L : 控制方向
R : 重置模型
空格 : 重置控制参数
```

运行示例

![screenshot](https://github.com/user-attachments/assets/2a03167c-a71e-4b63-84c3-6b3082c06f01)
