from typing import Dict
import torch
import numpy as np
import pandas as pd
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.common.replay_buffer import ReplayBuffer
from diffusion_policy_3d.common.sampler import (SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy_3d.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer, StringNormalizer
from diffusion_policy_3d.dataset.base_dataset import BaseDataset
import diffusion_policy_3d.model.vision_3d.point_process as point_process
from termcolor import cprint
import os

USE_STATE_QVEL = True
USE_STATE_BASE = True
USE_ACTION_QVEL = True

class MarsmindEpisodeDataset3D(BaseDataset):
    def __init__(self,
            data_path, 
            pad_before=0,
            pad_after=0,
            task_name=None,
            num_points=4096,
            ):
        super().__init__()
        cprint(f'Loading MarsmindEpisodeDataset from {data_path}/{task_name}', 'green')
        self.task_name = task_name

        self.num_points = num_points

        self.episode_state_data = []
        self.episode_action_data = []
        self.episode_pc_data = []

        self.episode_len = []

        for dir_episode in os.listdir(os.path.join(data_path, task_name)):
            episode_data = self.parse_episode(os.path.join(data_path, task_name, dir_episode))
            self.episode_state_data.append(episode_data['state'])
            self.episode_action_data.append(episode_data['action'])
            self.episode_pc_data.append(episode_data['pc'])

            self.episode_len.append(len(episode_data['state']))
        
        assert len(self.episode_state_data) == len(self.episode_action_data) and \
            len(self.episode_state_data) == len(self.episode_pc_data), 'data length error!'
        
        self.cumulative_len = np.cumsum(self.episode_len)
        self.train_idx = np.arange(-pad_before, pad_after + 1)
        self.obs_len = pad_before + 1

    def get_normalizer(self, mode='limits', **kwargs):
        data = {'action': np.concatenate(self.episode_action_data, axis=0)}
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)

        normalizer['point_cloud'] = SingleFieldLinearNormalizer.create_identity()
        normalizer['agent_pos'] = SingleFieldLinearNormalizer.create_identity()
        
        return normalizer

    def __len__(self) -> int:
        return self.cumulative_len[-1]

    def _sample_to_data(self, sample):
        agent_pos = sample['state'][:,].astype(np.float32)
        point_cloud = sample['point_cloud'][:,].astype(np.float32)
        # point_cloud = point_process.uniform_sampling_numpy(point_cloud, self.num_points)
        data = {
            'obs': {
                'agent_pos': agent_pos,
                'point_cloud': point_cloud,
                },
            'action': sample['action'].astype(np.float32)}
           
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        episode_index = np.argmax(self.cumulative_len > idx)
        start_step = idx - (self.cumulative_len[episode_index] - self.episode_len[episode_index])
        
        episode_state = self.episode_state_data[episode_index]
        episode_action = self.episode_action_data[episode_index]
        episode_pc = self.episode_pc_data[episode_index]

        train_idx = self.train_idx + start_step
        train_idx[train_idx < 0] = 0
        train_idx[train_idx > self.episode_len[episode_index] - 1] = self.episode_len[episode_index] - 1

        sample = dict()
        sample['state'] = np.array(episode_state)[train_idx[:self.obs_len]]
        sample['action'] = np.array(episode_action)[train_idx]
        sample['point_cloud'] = np.array(episode_pc)[train_idx[:self.obs_len]]

        data = self._sample_to_data(sample)
        to_torch_function = lambda x: torch.from_numpy(x) if x.__class__.__name__ == 'ndarray' else x
        torch_data = dict_apply(data, to_torch_function)
        return torch_data

    
    def parse_episode(self, file_path):
        metaact_list = []
        state = []
        action = []
        pc = []
        for dir_metaaction in os.listdir(file_path):
            if dir_metaaction == 'PHOTO' or 'README' in dir_metaaction or 'json' in dir_metaaction or 'pt' in dir_metaaction:
                continue
            metaact_list.append(os.path.join(file_path, dir_metaaction))
        metaact_list = sorted(metaact_list, key=lambda x: int(x.split('/')[-1].split('_')[0]))
        
        for metaact in metaact_list:
            metaact_dict = self.parse_metaaction(metaact)
            state += metaact_dict['state']
            action += metaact_dict['action']
            pc += metaact_dict['pc']
        
        return {
            "state": state,
            "action": action,
            "pc": pc,
        }
    
    def parse_metaaction(self, file_path):
        log_files = sorted(os.listdir(os.path.join(file_path, "logs")))
        num_steps = len(log_files)

        EPS = 1e-2
        first_idx = -1
        state = []
        action = []
        pc = []
        
        first_log_data = self.parse_log_file(os.path.join(file_path, "logs", log_files[0]))
        qpos_first = first_log_data['joint_states_single_msg']['position']

        for i in range(1, num_steps):
            log_data = self.parse_log_file(os.path.join(file_path, "logs", log_files[i]))
            
            action_qpos = log_data['joint_states_msg']['position']
            action_qvel = log_data['joint_states_msg']['velocity']
            action_base_vel_y = log_data['cmd_vel_msg']['linear_x']
            action_base_delta_ang = log_data['cmd_vel_msg']['angular_z']

            qpos = log_data['joint_states_single_msg']['position']
            qvel = log_data['joint_states_single_msg']['velocity']
            base_vel_y = log_data['bunker_status_msg']['linear_velocity']
            base_delta_ang = log_data['bunker_status_msg']['angular_velocity']

            if first_idx == -1:
                qpos_now = qpos
                qpos_delta = np.abs(qpos_now - qpos_first)

                if np.any(qpos_delta > EPS) or qpos_delta[-1] > EPS/2 or np.abs(base_delta_ang) >= EPS or np.abs(base_vel_y) >= EPS:
                    first_idx = i

                    first_log_data = self.parse_log_file(os.path.join(file_path, "logs", log_files[i - 1]))

                    action_qpos_first = first_log_data['joint_states_msg']['position']
                    action_qvel_first = first_log_data['joint_states_msg']['velocity']
                    action_base_vel_y_first = first_log_data['cmd_vel_msg']['linear_x']
                    action_base_delta_ang_first = first_log_data['cmd_vel_msg']['angular_z']

                    qpos_first = first_log_data['joint_states_single_msg']['position']
                    qvel_first = first_log_data['joint_states_single_msg']['velocity']
                    base_vel_y_first = first_log_data['bunker_status_msg']['linear_velocity']
                    base_delta_ang_first = first_log_data['bunker_status_msg']['angular_velocity']

                    state_list = []
                    action_list = []
                    state_list += qpos_first.tolist()
                    action_list += action_qpos_first.tolist()
                    if USE_STATE_QVEL:
                        state_list += qvel_first.tolist()
                    if USE_ACTION_QVEL:
                        action_list += action_qvel_first[:6].tolist()
                    if USE_STATE_BASE:
                        state_list += [base_vel_y_first]+[base_delta_ang_first]
                    action_list += [action_base_vel_y_first]+[action_base_delta_ang_first]

                    state.append(state_list)
                    action.append(action_list)

                    first_pc_data = pd.read_csv(os.path.join(file_path, "pc", log_files[i - 1].replace('.log', '.csv')))
                    first_pc_data = first_pc_data[['x', 'y', 'z']].to_numpy()
                    first_pc_data = point_process.uniform_sampling_numpy(first_pc_data[None, :], self.num_points)[0]
                    pc.append(first_pc_data)
                    
                else:
                    if i == num_steps - 1:
                        raise ValueError("Found no qpos that exceeds the threshold.")
                    continue

            state_list = []
            action_list = []
            state_list += qpos.tolist()
            action_list += action_qpos.tolist()
            if USE_STATE_QVEL:
                state_list += qvel.tolist()
            if USE_ACTION_QVEL:
                action_list += action_qvel[:6].tolist()
            if USE_STATE_BASE:
                state_list += [base_vel_y]+[base_delta_ang]
            action_list += [action_base_vel_y]+[action_base_delta_ang]
            
            state.append(state_list)
            action.append(action_list)

            pc_data = pd.read_csv(os.path.join(file_path, "pc", log_files[i].replace('.log', '.csv')))
            pc_data = pc_data[['x', 'y', 'z']].to_numpy()
            pc_data = point_process.uniform_sampling_numpy(pc_data[None, :], self.num_points)[0]
            pc.append(pc_data)

        # Return the resulting sample
        return {
            "state": state,
            "action": action,
            "pc": pc,
        }

    def parse_log_file(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            log_content = f.read()
        
        result = {}

        bunker_status_start = log_content.find("bunker_status_msg:")
        bunker_status_data = log_content[bunker_status_start:].split("cmd_vel_msg:")[0]
        linear_velocity = float(bunker_status_data.split("linear_velocity:")[1].split("\n")[0].strip())
        angular_velocity = float(bunker_status_data.split("angular_velocity:")[1].split("\n")[0].strip())
        result['bunker_status_msg'] = {'linear_velocity': linear_velocity, 'angular_velocity': angular_velocity}

        cmd_vel_start = log_content.find("cmd_vel_msg:")
        cmd_vel_data = log_content[cmd_vel_start:].split("joint_states_single_msg:")[0]
        linear_x = float(cmd_vel_data.split("Linear: x=")[1].split(",")[0].strip())
        angular_z = float(cmd_vel_data.split("Angular: x=0.0, y=0.0, z=")[1].split("\n")[0].strip())
        result['cmd_vel_msg'] = {'linear_x': linear_x, 'angular_z': angular_z}

        joint_states_single_start = log_content.find("joint_states_single_msg:")
        joint_states_single_data = log_content[joint_states_single_start:].split("end_pose_msg:")[0]
        position = joint_states_single_data.split("position:")[1].split("\n")[0].strip()
        velocity = joint_states_single_data.split("velocity:")[1].split("\n")[0].strip()
        result['joint_states_single_msg'] = {
            'position': np.array(eval(position)),
            'velocity': np.array(eval(velocity))
        }

        joint_states_start = log_content.find("joint_states_msg:")
        joint_states_data = log_content[joint_states_start:].split("effort:")[0]
        position = joint_states_data.split("position:")[1].split("\n")[0].strip()
        velocity = joint_states_data.split("velocity:")[1].split("\n")[0].strip()
        result['joint_states_msg'] = {
            'position': np.array(eval(position)),
            'velocity': np.array(eval(velocity))
        }

        return result


if __name__ == '__main__':
    dataset = MarsmindEpisodeDataset3D(
        data_path='/media/viewer/Image_Lab/embod_data/MarsMind_data',
        pad_after=15,
        pad_before=1,
        task_name='Sample',
        num_points=4096
    )
    print(len(dataset))