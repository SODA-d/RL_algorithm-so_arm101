import numpy as np
import mujoco
import gymnasium as gym 
from gymnasium import spaces
# from stable_baselines3 import PPO
# from stable_baselines3.common.env_util import make_vec_env
# from stable_baselines3.common.vec_env import SubprocVecEnv
import torch.nn as nn
import warnings
import torch
import mujoco.viewer
import time
from typing import Optional
from scipy.spatial.transform import Rotation as R

# 忽略stable-baselines3的冗余UserWarning
warnings.filterwarnings("ignore", category=UserWarning, module="stable_baselines3.common.on_policy_algorithm")

import os

def write_flag_file(flag_filename="rl_visu_flag"):
    flag_path = os.path.join("/tmp", flag_filename)
    try:
        with open(flag_path, "w") as f:
            f.write("This is a flag file")
        return True
    except Exception as e:
        return False

def check_flag_file(flag_filename="rl_visu_flag"):
    flag_path = os.path.join("/tmp", flag_filename)
    return os.path.exists(flag_path)

def delete_flag_file(flag_filename="rl_visu_flag"):
    flag_path = os.path.join("/tmp", flag_filename)
    if not os.path.exists(flag_path):
        return True
    try:
        os.remove(flag_path)
        return True
    except Exception as e:
        return False

class PandaObstacleEnv(gym.Env):
    def __init__(self, visualize: bool = False):
        super(PandaObstacleEnv, self).__init__()
        if not check_flag_file():
            write_flag_file()
            self.visualize = visualize
        else:
            self.visualize = False
        self.handle = None

        self.model = mujoco.MjModel.from_xml_path('trs_so_arm101/scene.xml')
        self.data = mujoco.MjData(self.model)
        
        if self.visualize:
            self.handle = mujoco.viewer.launch_passive(self.model, self.data)
            self.handle.cam.distance = 3.0
            self.handle.cam.azimuth = 0.0
            self.handle.cam.elevation = -30.0
            self.handle.cam.lookat = np.array([0.2, 0.0, 0.4])
        
        self.end_effector_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'gripper')
        self.initial_ee_pos = np.zeros(3, dtype=np.float32) 
        self.home_joint_pos = np.array([  # home位姿
            0.0, 0.0, 0.0,  
            0.0, 0.0, 0.0
        ], dtype=np.float32)
        
        self.goal_size = 0.03
        
        # 约束工作空间
        self.workspace = {
            'x': [0.2, 0.25],
            'y': [-0.25, 0.25],
            'z': [0.1, 0.35]
        }
        
        # 动作空间与观测空间
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        # 7轴关节角度、目标位置
        self.obs_size = 6 + 3 
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_size,), dtype=np.float32)
        
        self.goal = np.zeros(3, dtype=np.float32)
        self.np_random = np.random.default_rng(None)
        self.prev_action = np.zeros(6, dtype=np.float32)
        self.goal_threshold = 0.005

    def _get_valid_goal(self) -> np.ndarray:
        """生成有效目标点"""
        while True:
            goal = self.np_random.uniform(
                low=[self.workspace['x'][0], self.workspace['y'][0], self.workspace['z'][0]],
                high=[self.workspace['x'][1], self.workspace['y'][1], self.workspace['z'][1]]
            )
            if 0.1 < np.linalg.norm(goal - self.initial_ee_pos) < 0.3 and goal[0] > 0.2 and goal[2] > 0.1:
                return goal.astype(np.float32)

    def _render_scene(self) -> None:
        """渲染目标点"""
        if not self.visualize or self.handle is None:
            return
        self.handle.user_scn.ngeom = 0
        total_geoms = 1
        self.handle.user_scn.ngeom = total_geoms

        # 渲染目标点（蓝色）
        goal_rgba = np.array([0.1, 0.1, 0.9, 0.9], dtype=np.float32)
        mujoco.mjv_initGeom(
            self.handle.user_scn.geoms[0],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[self.goal_size, 0.0, 0.0],
            pos=self.goal,
            mat=np.eye(3).flatten(),
            rgba=goal_rgba
        )

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        
        # 重置关节到home位姿
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:6] = self.home_joint_pos
        mujoco.mj_forward(self.model, self.data)
        self.initial_ee_pos = self.data.body(self.end_effector_id).xpos.copy()
        self.start_ee_pos = self.initial_ee_pos.copy()
        
        # 生成目标
        self.goal = self._get_valid_goal()
        if self.visualize:
            self._render_scene()        
        
        obs = self._get_observation()
        self.start_t = time.time()
        return obs, {}

    def _get_observation(self) -> np.ndarray:
        joint_pos = self.data.qpos[:6].copy().astype(np.float32)
        # ee_pos = self.data.body(self.end_effector_id).xpos.copy().astype(np.float32)
        # ee_quat = self.data.body(self.end_effector_id).xquat.copy().astype(np.float32)
        return np.concatenate([joint_pos, self.goal])

    # def _calc_reward(self, ee_pos: np.ndarray, ee_orient: np.ndarray, joint_angles: np.ndarray, action: np.ndarray) -> tuple[np.ndarray, float]:
    #     dist_to_goal = np.linalg.norm(ee_pos - self.goal)
        
    #     # 非线性距离奖励
    #     if dist_to_goal < self.goal_threshold:
    #         distance_reward = 100.0
    #     elif dist_to_goal < 2*self.goal_threshold:
    #         distance_reward = 50.0
    #     elif dist_to_goal < 3*self.goal_threshold:
    #         distance_reward = 10.0
    #     else:
    #         distance_reward = 1.0 / (1.0 + dist_to_goal)

    #     # 计算起点到目标的向量
    #     start_to_goal = self.goal - self.start_ee_pos
    #     start_to_goal_norm = np.linalg.norm(start_to_goal)
    #     if start_to_goal_norm < 1e-6:  # 避免除以0（理论上不会发生，因目标与起点有距离约束）
    #         linearity_penalty = 0.0
    #     else:
    #         # 计算当前位置到起点的向量
    #         start_to_current = ee_pos - self.start_ee_pos
    #         # 计算当前位置在“起点→目标”直线上的投影比例（0~1之间表示在两点之间）
    #         projection_ratio = np.dot(start_to_current, start_to_goal) / (start_to_goal_norm **2)
    #         projection_ratio = np.clip(projection_ratio, 0.0, 1.0)  # 限制在0~1范围（超出目标点后不再惩罚）
    #         # 计算直线上的投影点
    #         projected_point = self.start_ee_pos + projection_ratio * start_to_goal
    #         # 计算当前位置与投影点的垂直距离（偏离直线的程度）
    #         linearity_error = np.linalg.norm(ee_pos - projected_point)
    #         # 直线性惩罚（距离越大，惩罚越重）
    #         linearity_penalty = 0.7 * linearity_error  # 权重可根据需要调整
                
    #     # 姿态约束：保持末端朝下
    #     target_orient = np.array([0, 0, -1])
    #     ee_orient_norm = ee_orient / np.linalg.norm(ee_orient)
    #     dot_product = np.dot(ee_orient_norm, target_orient)
    #     angle_error = np.arccos(np.clip(dot_product, -1.0, 1.0))
    #     orientation_penalty = 0.3 * angle_error
        
    #     # 动作相关惩罚
    #     action_diff = action - self.prev_action
    #     smooth_penalty = 0.1 * np.linalg.norm(action_diff)
    #     action_magnitude_penalty = 0.05 * np.linalg.norm(action)

    #     contact_reward = 1.0*self.data.ncon
        
    #     # 关节角度限制惩罚
    #     joint_penalty = 0.0
    #     for i in range(7):
    #         min_angle, max_angle = self.model.jnt_range[:7][i]
    #         if joint_angles[i] < min_angle:
    #             joint_penalty += 0.5 * (min_angle - joint_angles[i])
    #         elif joint_angles[i] > max_angle:
    #             joint_penalty += 0.5 * (joint_angles[i] - max_angle)
        
    #     # 时间惩罚
    #     time_penalty = 0.01
        
    #     # v1
    #     # total_reward = distance_reward - contact_reward - smooth_penalty - orientation_penalty       
    #     # v2
    #     # total_reward = distance_reward - contact_reward - smooth_penalty - orientation_penalty - linearity_penalty
    #     # v3
    #     total_reward = distance_reward - contact_reward - smooth_penalty - orientation_penalty - joint_penalty
    #     # print(f"[奖励] 距离目标: {distance_reward:.3f}, [碰撞]: {contact_reward:.3f}, 动作惩罚: {smooth_penalty:.3f}, 姿态: {orientation_penalty:.3f}  总奖励: {total_reward:.3f}")
        
    #     # 更新上一步动作
    #     self.prev_action = action.copy()
        
    #     return total_reward, dist_to_goal, angle_error

    def _calc_reward(self, ee_pos: np.ndarray, ee_orient: np.ndarray, joint_angles: np.ndarray, action: np.ndarray) -> tuple[np.ndarray, float]:
        dist_to_goal = np.linalg.norm(ee_pos - self.goal)
    
        # 非线性距离奖励（保持不变）_get_valid_goal
        if dist_to_goal < self.goal_threshold:
            distance_reward = 100.0
        elif dist_to_goal < 2*self.goal_threshold:
            distance_reward = 50.0
        elif dist_to_goal < 3*self.goal_threshold:
            distance_reward = 10.0
        else:
            distance_reward = 1.0 / (1.0 + dist_to_goal)

        # 计算起点到目标的向量及相关参数
        start_to_goal = self.goal - self.start_ee_pos
        start_to_goal_norm = np.linalg.norm(start_to_goal)
        linearity_reward = 0.0
        deviation_penalty = 0.0
        
        if start_to_goal_norm >= 1e-6:  # 起点和目标不重合时才计算直线相关奖励/惩罚
            # 计算当前位置到起点的向量
            start_to_current = ee_pos - self.start_ee_pos
            # 计算当前位置在“起点→目标”直线上的投影比例（限制在0~1，避免超出目标后惩罚）
            projection_ratio = np.dot(start_to_current, start_to_goal) / (start_to_goal_norm **2)
            projection_ratio = np.clip(projection_ratio, 0.0, 1.0)
            # 计算直线上的投影点，得到当前位置偏离直线的垂直距离
            projected_point = self.start_ee_pos + projection_ratio * start_to_goal
            linearity_error = np.linalg.norm(ee_pos - projected_point)  # 偏离直线的距离
            
            # 1. 直线接近奖励：离直线越近，奖励越高（非线性递增）
            linearity_reward = 3.0 / (1.0 + linearity_error)  # 系数8.0可根据重要性调整
            
            # 2. 远离趋势惩罚：检测“先靠近后远离”的行为
            # 初始化或更新历史最小偏离距离（跟踪最近点）
            if not hasattr(self, 'min_linearity_error'):
                self.min_linearity_error = np.inf  # 首次运行初始化
            if linearity_error < self.min_linearity_error:
                self.min_linearity_error = linearity_error  # 更近时更新最小值，无惩罚
            else:
                # 比最近点更远时，惩罚远离的程度（距离差越大，惩罚越重）
                deviation_penalty = 1.0 * (linearity_error - self.min_linearity_error)  # 系数3.0可调整

        # 姿态约束：保持末端朝下（保持不变）
        target_orient = np.array([0, 0, 1])
        ee_orient_norm = ee_orient / np.linalg.norm(ee_orient)
        dot_product = np.dot(ee_orient_norm, target_orient)
        angle_error = np.arccos(np.clip(dot_product, -1.0, 1.0))
        orientation_penalty = 0.3 * angle_error
        
        # 动作相关惩罚（保持不变）
        # action_diff = action - self.prev_action
        # smooth_penalty = 0.1 * np.linalg.norm(action_diff)
        # action_magnitude_penalty = 0.05 * np.linalg.norm(action)

        # 碰撞惩罚（保持不变）
        # contact_reward = 1.0 * self.data.ncon
        
        # 关节角度限制惩罚（保持不变）
        joint_penalty = 0.0
        for i in range(6):
            min_angle, max_angle = self.model.jnt_range[:6][i]
            if joint_angles[i] < min_angle:
                joint_penalty += 0.5 * (min_angle - joint_angles[i])
            elif joint_angles[i] > max_angle:
                joint_penalty += 0.5 * (joint_angles[i] - max_angle)
        
        # 时间惩罚（保持不变）
        time_penalty = 0.01
        
        # 总奖励：整合新的直线奖励和远离惩罚
        total_reward = (distance_reward 
                    # + linearity_reward  # 新增：靠近直线的奖励
                    # - contact_reward 
                    # - smooth_penalty 
                    # - orientation_penalty 
                    - joint_penalty 
                    )  # 新增：先近后远的惩罚
        
        # 更新上一步动作
        self.prev_action = action.copy()
        total_reward = np.float32(total_reward)  
        return total_reward, dist_to_goal, angle_error

    def step(self, action: np.ndarray) -> tuple[np.ndarray, np.float32, bool, bool, dict]:
        # 动作缩放
        joint_ranges = self.model.jnt_range[:6]
        scaled_action = np.zeros(6, dtype=np.float32)
        for i in range(6):
            scaled_action[i] = joint_ranges[i][0] + (action[i] + 1) * 0.5 * (joint_ranges[i][1] - joint_ranges[i][0])
        
        # 执行动作
        self.data.ctrl[:6] = scaled_action
        mujoco.mj_step(self.model, self.data)
        
        # 计算奖励与状态
        ee_pos = self.data.body(self.end_effector_id).xpos.copy()
        ee_quat = self.data.body(self.end_effector_id).xquat.copy()
        rot = R.from_quat(ee_quat)
        ee_quat_euler_rad = rot.as_euler('xyz')
        reward, dist_to_goal,_ = self._calc_reward(ee_pos, ee_quat_euler_rad, self.data.qpos[:6], action)
        terminated = False
        collision = False
        
        # 目标达成
        if dist_to_goal < self.goal_threshold:
            terminated = True
        # print(f"[奖励] 距离目标: {dist_to_goal:.3f}, 奖励: {reward:.3f}")

        if not terminated:
            if time.time() - self.start_t > 20.0:
                reward -= 10.0
                print(f"[超时] 时间过长，奖励减半")
                terminated = True

        if self.visualize and self.handle is not None:
            self.handle.sync()
            time.sleep(0.01) 
        
        obs = self._get_observation()
        info = {
            'is_success': terminated and (dist_to_goal < self.goal_threshold),
            'distance_to_goal': dist_to_goal,
            'collision': collision
        }
        
        return obs, reward.astype(np.float32), terminated, False, info

    def seed(self, seed: Optional[int] = None) -> list[Optional[int]]:
        self.np_random = np.random.default_rng(seed)
        return [seed]

    def close(self) -> None:
        if self.visualize and self.handle is not None:
            self.handle.close()
            self.handle = None
        print("环境已关闭，资源释放完成")


def train_ppo(
    n_envs: int = 24,
    total_timesteps: int = 40_000_000,  # 本次训练的新增步数
    model_save_path: str = "panda_ppo_reach_target",
    visualize: bool = False,
    resume_from: Optional[str] = None
) -> None:

    ENV_KWARGS = {'visualize': visualize}
    
    env = make_vec_env(
        env_id=lambda: PandaObstacleEnv(** ENV_KWARGS),
        n_envs=n_envs,
        seed=42,
        vec_env_cls=SubprocVecEnv,
        vec_env_kwargs={"start_method": "fork"}
    )
    
    if resume_from is not None:
        model = PPO.load(resume_from, env=env)  # 加载时需传入当前环境
    else:
        POLICY_KWARGS = dict(
            activation_fn=nn.ReLU,
            net_arch=[dict(pi=[256, 128], vf=[256, 128])]
        )
        model = PPO(
            policy="MlpPolicy",
            env=env,
            policy_kwargs=POLICY_KWARGS,
            verbose=1,
            n_steps=2048,          
            batch_size=2048,       
            n_epochs=10,           
            gamma=0.99,
            learning_rate=2e-4,
            device="cuda" if torch.cuda.is_available() else "cpu",
            tensorboard_log="./tensorboard/panda_reach_target/"
        )
    
    print(f"并行环境数: {n_envs}, 本次训练新增步数: {total_timesteps}")
    model.learn(
        total_timesteps=total_timesteps,
        progress_bar=True
    )
    
    model.save(model_save_path)
    env.close()
    print(f"模型已保存至: {model_save_path}")


def test_ppo(
    model_path: str = "panda_ppo_reach_target",
    total_episodes: int = 5,
) -> None:
    env = PandaObstacleEnv(visualize=True)
    model = PPO.load(model_path, env=env)
    
    record_gif = False
    frames = [] if record_gif else None
    render_scene = None  
    render_context = None 
    pixel_buffer = None 
    viewport = None
    
    success_count = 0
    print(f"测试轮数: {total_episodes}")
    
    for ep in range(total_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0.0
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        
        if info['is_success']:
            success_count += 1
        print(f"轮次 {ep+1:2d} | 总奖励: {episode_reward:6.2f} | 结果: {'成功' if info['is_success'] else '碰撞/失败'}")
    
    success_rate = (success_count / total_episodes) * 100
    print(f"总成功率: {success_rate:.1f}%")
    
    env.close()


import torch
import torch.nn as nn
import gymnasium as gym 
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class CriticNet(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,1)
        )

    def forward(self, x):
        value = self.network(x)
        return value

class ActorNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.mu_head = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        
        self.std_head = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Softplus()
        )
    
    def forward(self, x):
        features = self.shared(x)
        mu = 2.0 * self.mu_head(features)
        std = self.std_head(features) + 1e-6  # 数值稳定性
        return mu, std
    
    


class PPO():
    def __init__(self, state_dim, hidden_dim, action_dim, lr, lmbda, epochs, eps, gamma, device):
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.lr = lr
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
        self.gamma = gamma
        self.device = device
        
        self.actor = ActorNet(state_dim=self.state_dim,hidden_dim=self.hidden_dim, action_dim=self.action_dim).to(self.device)
        self.critic = CriticNet(state_dim=self.state_dim, hidden_dim=self.hidden_dim).to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=5e-3)

    def take_action(self, state):
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        mu, std = self.actor(state)
        action_dist = torch.distributions.Normal(mu, std)
        action = action_dist.sample()
        return action.cpu().numpy().tolist()
    
    def update_network(self, transiton_dict):
        states = torch.tensor(np.array(transiton_dict['states']), dtype=torch.float, device=self.device)
        actions = torch.tensor(np.array(transiton_dict['actions']), dtype=torch.float, device=self.device)
        rewards = torch.tensor(np.array(transiton_dict['rewards']),dtype=torch.float, device=self.device).view(-1,1)
        next_states = torch.tensor(np.array(transiton_dict['next_states']), dtype=torch.float, device=self.device)
        dones = torch.tensor(np.array(transiton_dict['dones']), dtype=torch.float,device=self.device).view(-1,1)

        #计算优势函数
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_error = td_target - self.critic(states)
        advantage = self.compute_advantage(gamma=self.gamma, lmbda=self.lmbda, td_error=td_error)


        #重要性采样：动作在旧策略的概率
        mu, std = self.actor(states)
        action_dists = torch.distributions.Normal(mu.detach(), std.detach())
        old_log_probs = action_dists.log_prob(actions)

        for _ in range(self.epochs):
            mu, std = self.actor(states)
            action_dists = torch.distributions.Normal(mu, std)
            new_log_probs = action_dists.log_prob(actions)#同样的动作在更新了参数后的对数概率
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio*advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage
            actor_loss = torch.mean(-torch.min(surr1, surr2))
            critic_loss = nn.MSELoss()(self.critic(states), td_target.detach())
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()
      


    def compute_advantage(self, gamma, lmbda, td_error):
        td_error = td_error.detach().cpu().numpy()
        advantage_list = []
        advantage = 0.0
        for delta in td_error[::-1]:
            advantage = gamma * lmbda * advantage + delta
            advantage_list.append(advantage)
        advantage_list.reverse()
        return torch.tensor(advantage_list, dtype=torch.float, device=self.device)





if __name__ == "__main__":
    lr = 1e-4
    num_episodes = 2000
    hidden_dim = 128
    gamma = 0.9
    lmbda = 0.9
    epochs = 10
    eps = 0.2
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
        "cpu")
    from gymnasium.envs.registration import register

    register(
        id='CustomEnv-v0',
        entry_point=__name__ + ':PandaObstacleEnv',
    )
    env_name = 'CustomEnv-v0'
    env = gym.make(env_name)
    torch.manual_seed(0)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]  # 连续动作空间
    agent = PPO(state_dim, hidden_dim, action_dim,lr, lmbda, epochs, eps, gamma, device)


    return_list = []

    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                state,_ = env.reset(seed=42)
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, terminated, truncated,_ = env.step(action)
                    done = terminated or truncated
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    state = next_state
                    episode_return += reward
                return_list.append(episode_return)
                agent.update_network(transition_dict)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)