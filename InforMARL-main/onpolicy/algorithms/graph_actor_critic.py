import argparse
from typing import Tuple, List

import gym
import torch
from torch import Tensor
import torch.nn as nn
from onpolicy.algorithms.utils.util import init, check
from onpolicy.algorithms.utils.gnn import GNNBase
from onpolicy.algorithms.utils.mlp import MLPBase
from onpolicy.algorithms.utils.rnn import RNNLayer
from onpolicy.algorithms.utils.act import ACTLayer
from onpolicy.algorithms.utils.popart import PopArt
from onpolicy.utils.util import get_shape_from_obs_space

import numpy as np#檢查用

def minibatchGenerator(
    obs: Tensor, node_obs: Tensor, adj: Tensor, agent_id: Tensor, max_batch_size: int
):
    """
    Split a big batch into smaller batches.
    """
    num_minibatches = obs.shape[0] // max_batch_size + 1
    for i in range(num_minibatches):
        yield (
            obs[i * max_batch_size : (i + 1) * max_batch_size],
            node_obs[i * max_batch_size : (i + 1) * max_batch_size],
            adj[i * max_batch_size : (i + 1) * max_batch_size],
            agent_id[i * max_batch_size : (i + 1) * max_batch_size],
        )


class GR_Actor(nn.Module):
    """
    Actor network class for MAPPO. Outputs actions given observations.
    args: argparse.Namespace
        Arguments containing relevant model information.
    obs_space: (gym.Space)
        Observation space.
    node_obs_space: (gym.Space)
        Node observation space
    edge_obs_space: (gym.Space)
        Edge dimension in graphs
    action_space: (gym.Space)
        Action space.
    device: (torch.device)
        Specifies the device to run on (cpu/gpu).
    split_batch: (bool)
        Whether to split a big-batch into multiple
        smaller ones to speed up forward pass.
    max_batch_size: (int)
        Maximum batch size to use.
    """

    def __init__(
        self,
        args: argparse.Namespace,
        obs_space: gym.Space,
        node_obs_space: gym.Space,
        edge_obs_space: gym.Space,
        action_space: gym.Space,
        device=torch.device("cpu"),
        split_batch: bool = False,
        max_batch_size: int = 32,
    ) -> None:
        super(GR_Actor, self).__init__()
        self.args = args
        self.hidden_size = args.hidden_size

        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self.split_batch = split_batch
        self.max_batch_size = max_batch_size
        self.tpdv = dict(dtype=torch.float32, device=device)
        #add1
        self.action_space = action_space
        obs_shape = get_shape_from_obs_space(obs_space)
        node_obs_shape = get_shape_from_obs_space(node_obs_space)[
            1
        ]  # returns (num_nodes, num_node_feats)
        edge_dim = get_shape_from_obs_space(edge_obs_space)[0]  # returns (edge_dim,)

        self.gnn_base = GNNBase(args, node_obs_shape, edge_dim, args.actor_graph_aggr)
        gnn_out_dim = self.gnn_base.out_dim  # output shape from gnns
        mlp_base_in_dim = gnn_out_dim + obs_shape[0]
        self.base = MLPBase(args, obs_shape=None, override_obs_dim=mlp_base_in_dim)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(
                self.hidden_size,
                self.hidden_size,
                self._recurrent_N,
                self._use_orthogonal,
            )

        self.act = ACTLayer(
            action_space, self.hidden_size, self._use_orthogonal, self._gain
        )

        self.to(device)
        #add1
        # 添加監測統計數據的屬性
        self.stats = {
            'rank_distribution': [],           # 記錄排名分布
            'action_mask_stats': [],          # 記錄動作遮罩統計
            'selected_actions': [],           # 記錄被選擇的動作
            'coverage_scores': [],            # 記錄覆蓋分數
            'restriction_levels': [],         # 記錄限制程度
        }
        #add2
    
    def forward(
        self,
        obs,
        node_obs,
        adj,
        agent_id,
        rnn_states,
        masks,
        available_actions=None,
        deterministic=False,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compute actions from the given inputs.
        obs: (np.ndarray / torch.Tensor)
            Observation inputs into network.
        node_obs (np.ndarray / torch.Tensor):
            Local agent graph node features to the actor.
        adj (np.ndarray / torch.Tensor):
            Adjacency matrix for the graph
        agent_id (np.ndarray / torch.Tensor)
            The agent id to which the observation belongs to
        rnn_states: (np.ndarray / torch.Tensor)
            If RNN network, hidden states for RNN.
        masks: (np.ndarray / torch.Tensor)
            Mask tensor denoting if hidden states
            should be reinitialized to zeros.
        available_actions: (np.ndarray / torch.Tensor)
            Denotes which actions are available to agent
            (if None, all actions available)
        deterministic: (bool)
            Whether to sample from action distribution or return the mode.

        :return actions: (torch.Tensor)
            Actions to take.
        :return action_log_probs: (torch.Tensor)
            Log probabilities of taken actions.
        :return rnn_states: (torch.Tensor)
            Updated RNN hidden states.
        """
        obs = check(obs).to(**self.tpdv)
        node_obs = check(node_obs).to(**self.tpdv)
        adj = check(adj).to(**self.tpdv)
        agent_id = check(agent_id).to(**self.tpdv).long()
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        #add1
        ranks = self.calculate_coverage_ranks(node_obs)
        
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        # if batch size is big, split into smaller batches, forward pass and then concatenate
        if (self.split_batch) and (obs.shape[0] > self.max_batch_size):
            # print(f'Actor obs: {obs.shape[0]}')
            batchGenerator = minibatchGenerator(
                obs, node_obs, adj, agent_id, self.max_batch_size
            )
            actor_features = []
            for batch in batchGenerator:
                obs_batch, node_obs_batch, adj_batch, agent_id_batch = batch
                nbd_feats_batch = self.gnn_base(
                    node_obs_batch, adj_batch, agent_id_batch
                )
                act_feats_batch = torch.cat([obs_batch, nbd_feats_batch], dim=1)
                actor_feats_batch = self.base(act_feats_batch)
                actor_features.append(actor_feats_batch)
            actor_features = torch.cat(actor_features, dim=0)
        else:
            nbd_features = self.gnn_base(node_obs, adj, agent_id)
            actor_features = torch.cat([obs, nbd_features], dim=1)
            actor_features = self.base(actor_features)
        
        #add1
        action_masks = self.create_priority_based_mask(node_obs, agent_id, ranks, actor_features)
        if available_actions is None:
            available_actions = torch.ones(self.action_space.n).to(**self.tpdv)
        available_actions = available_actions * action_masks
        #add2
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)
            
        actions, action_log_probs = self.act(
            actor_features, available_actions, deterministic
        )
        return (actions, action_log_probs, rnn_states)
    #add1
    def calculate_coverage_ranks(self, node_obs):
        """
        計算每個智能體的覆蓋排名，考慮更多因素
        """
        batch_size = node_obs.shape[0]
        num_nodes = node_obs.shape[1]
        batch_ranks = []
        batch_scores = []
        batch_distances = []
        
        for batch_idx in range(batch_size):
            coverage_scores = []
            for node_id in range(num_nodes):
                # 計算目標數量
                goal_count = (node_obs[batch_idx, node_id, -1] == 2).sum().item()
                
                # 計算到目標的平均距離
                goal_positions = node_obs[batch_idx, node_obs[batch_idx, :, -1] == 2, :2]
                if len(goal_positions) > 0:
                    agent_pos = node_obs[batch_idx, node_id, :2]
                    distances = torch.norm(goal_positions - agent_pos, dim=1)
                    avg_distance = distances.mean().item()
                else:
                    avg_distance = float('inf')
                
                # 綜合評分（目標數量越多越好，距離越近越好）
                score = goal_count - 0.1 * avg_distance  # 可調整權重
                coverage_scores.append((node_id, score))
            
            # 根據綜合評分排序
            sorted_agents = sorted(coverage_scores, key=lambda x: x[1], reverse=True)
            batch_rank = [-1] * num_nodes
            for rank, (node_id, _) in enumerate(sorted_agents):
                batch_rank[node_id] = rank
            batch_ranks.append(batch_rank)
            
            # 收集統計數據
            self.stats['rank_distribution'].append(batch_rank)
            self.stats['coverage_scores'].append([score for _, score in coverage_scores])
        
        return torch.tensor(batch_ranks).to(**self.tpdv)
    
    def create_priority_based_mask(self, node_obs, agent_id, ranks, actor_features):
        """
        使用動作概率預測的優先級遮罩策略
        """
        batch_size = node_obs.shape[0]
        action_masks = torch.ones(batch_size, self.action_space.n).to(**self.tpdv)
        
        if not torch.is_tensor(ranks):
            ranks = torch.tensor(ranks).to(**self.tpdv)
        
        # 獲取所有智能體的動作概率分布
        all_action_probs = self.act.get_probs(actor_features)  # [batch_size, num_actions]
        
        for batch_idx in range(batch_size):
            current_agent = agent_id[batch_idx].item()
            current_rank = ranks[batch_idx][current_agent].item()
            
            if current_rank > 0:  # 非最高優先級的智能體
                current_pos = node_obs[batch_idx, current_agent, :2]
                
                # 找出所有優先級更高的智能體
                higher_priority_positions = []
                higher_priority_probs = []
                
                for other_agent in range(node_obs.shape[1]):
                    other_rank = ranks[batch_idx][other_agent].item()
                    if other_rank < current_rank:  # 優先級更高的智能體
                        other_pos = node_obs[batch_idx, other_agent, :2]
                        other_probs = all_action_probs[batch_idx]  # 該智能體的動作概率
                        higher_priority_positions.append(other_pos)
                        higher_priority_probs.append(other_probs)
                
                if higher_priority_positions:
                    # 創建動作遮罩
                    mask = torch.ones(self.action_space.n).to(**self.tpdv)
                    
                    # 計算基礎限制程度
                    restriction_level = min(current_rank / (node_obs.shape[1] - 1), 0.8)
                    
                    # 對每個可能的動作進行評估
                    for action_idx in range(1, self.action_space.n):  # 跳過停留動作
                        next_pos = self.calculate_next_position(current_pos, action_idx)
                        
                        # 檢查與所有高優先級智能體的預測動作的衝突
                        for high_pos, high_probs in zip(higher_priority_positions, higher_priority_probs):
                            # 獲取最可能的動作（概率前3高的動作）
                            _, top_actions = torch.topk(high_probs, k=3)
                            
                            for pred_action in top_actions:
                                pred_next_pos = self.calculate_next_position(high_pos, pred_action.item())
                                distance = torch.norm(next_pos - pred_next_pos)
                                
                                # 根據距離和動作概率調整遮罩值
                                if distance < 0.5:  # 安全距離閾值
                                    # 使用動作概率來調整遮罩強度
                                    mask_reduction = high_probs[pred_action] * restriction_level
                                    mask[action_idx] *= (1 - mask_reduction)
                    
                    # 確保停留動作始終可用
                    mask[0] = 1.0
                    action_masks[batch_idx] = mask
                    
                    # 收集統計數據
                    mask_stats = {
                        'total_actions': action_masks.shape[1],
                        'allowed_actions': mask.sum().item(),
                        'restriction_level': restriction_level
                    }
                    self.stats['action_mask_stats'].append(mask_stats)
                    self.stats['restriction_levels'].append(restriction_level)
        
        return action_masks

    def calculate_next_position(self, current_pos, action_idx):
        """
        根據動作計算下一個位置
        """
        # 定義動作到方向的映射
        action_to_direction = {
            1: torch.tensor([0, 1]),    # 上
            2: torch.tensor([0, -1]),   # 下
            3: torch.tensor([-1, 0]),   # 左
            4: torch.tensor([1, 0]),    # 右
            5: torch.tensor([1, 1]),    # 右上
            6: torch.tensor([1, -1]),   # 右下
            7: torch.tensor([-1, 1]),   # 左上
            8: torch.tensor([-1, -1]),  # 左下
        }
        
        if action_idx in action_to_direction:
            direction = action_to_direction[action_idx].to(**self.tpdv)
            return current_pos + 0.1 * direction  # 0.1 是移動步長
        
        return current_pos
	
    def log_stats(self, episode=None):
        """輸出統計數據"""
        if episode is not None and episode % self.args.log_interval == 0:  # 使用 log_interval
            stats_summary = {}
            
            # 處理排名分布
            if len(self.stats['rank_distribution']) > 0:
                stats_summary['rank_dist'] = np.mean(self.stats['rank_distribution'][-1000:], axis=0)
            
            # 處理動作遮罩統計
            if len(self.stats['action_mask_stats']) > 0:
                # 提取 allowed_actions 的值並計算平均值
                allowed_actions = [stat['allowed_actions'] for stat in self.stats['action_mask_stats'][-1000:]]
                stats_summary['mask_ratio'] = np.mean(allowed_actions)
            
            # 處理選擇的動作
            if len(self.stats['selected_actions']) > 0:
                stats_summary['action_dist'] = np.bincount(self.stats['selected_actions'][-1000:])
            
            # 處理覆蓋分數
            if len(self.stats['coverage_scores']) > 0:
                stats_summary['avg_coverage'] = np.mean(self.stats['coverage_scores'][-1000:])
            
            # 處理限制程度
            if len(self.stats['restriction_levels']) > 0:
                stats_summary['avg_restriction'] = np.mean(self.stats['restriction_levels'][-1000:])
            
            
            
            print(f"\nEpisode {episode} Statistics:")
            for key, value in stats_summary.items():
                if isinstance(value, np.ndarray):
                    print(f"{key}: {value.tolist()}")
                else:
                    print(f"{key}: {value}")
            
            # 清空統計數據以節省內存
            for key in self.stats:
                self.stats[key] = []
                
            return stats_summary
    #add2

    def evaluate_actions(
        self,
        obs,
        node_obs,
        adj,
        agent_id,
        rnn_states,
        action,
        masks,
        available_actions=None,
        active_masks=None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute log probability and entropy of given actions.
        obs: (torch.Tensor)
            Observation inputs into network.
        node_obs (torch.Tensor):
            Local agent graph node features to the actor.
        adj (torch.Tensor):
            Adjacency matrix for the graph.
        agent_id (np.ndarray / torch.Tensor)
            The agent id to which the observation belongs to
        action: (torch.Tensor)
            Actions whose entropy and log probability to evaluate.
        rnn_states: (torch.Tensor)
            If RNN network, hidden states for RNN.
        masks: (torch.Tensor)
            Mask tensor denoting if hidden states
            should be reinitialized to zeros.
        available_actions: (torch.Tensor)
            Denotes which actions are available to agent
            (if None, all actions available)
        active_masks: (torch.Tensor)
            Denotes whether an agent is active or dead.

        :return action_log_probs: (torch.Tensor)
            Log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor)
            Action distribution entropy for the given inputs.
        """
        obs = check(obs).to(**self.tpdv)
        node_obs = check(node_obs).to(**self.tpdv)
        adj = check(adj).to(**self.tpdv)
        agent_id = check(agent_id).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        
            
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        # if batch size is big, split into smaller batches, forward pass and then concatenate
        if (self.split_batch) and (obs.shape[0] > self.max_batch_size):
            # print(f'eval Actor obs: {obs.shape[0]}')
            batchGenerator = minibatchGenerator(
                obs, node_obs, adj, agent_id, self.max_batch_size
            )
            actor_features = []
            for batch in batchGenerator:
                obs_batch, node_obs_batch, adj_batch, agent_id_batch = batch
                nbd_feats_batch = self.gnn_base(
                    node_obs_batch, adj_batch, agent_id_batch
                )
                act_feats_batch = torch.cat([obs_batch, nbd_feats_batch], dim=1)
                actor_feats_batch = self.base(act_feats_batch)
                actor_features.append(actor_feats_batch)
            actor_features = torch.cat(actor_features, dim=0)
        else:
            nbd_features = self.gnn_base(node_obs, adj, agent_id)
            actor_features = torch.cat([obs, nbd_features], dim=1)
            actor_features = self.base(actor_features)
            
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)
        #print("actor_features in actor268 NaN: ",torch.isnan(actor_features).sum().item())
        action_log_probs, dist_entropy = self.act.evaluate_actions(
            actor_features,
            action,
            available_actions,
            active_masks=active_masks if self._use_policy_active_masks else None,
        )

        return (action_log_probs, dist_entropy)


class GR_Critic(nn.Module):
    """
    Critic network class for MAPPO. Outputs value function predictions
    given centralized input (MAPPO) or local observations (IPPO).
    args: (argparse.Namespace)
        Arguments containing relevant model information.
    cent_obs_space: (gym.Space)
        (centralized) observation space.
    node_obs_space: (gym.Space)
        node observation space.
    edge_obs_space: (gym.Space)
        edge observation space.
    device: (torch.device)
        Specifies the device to run on (cpu/gpu).
    split_batch: (bool)
        Whether to split a big-batch into multiple
        smaller ones to speed up forward pass.
    max_batch_size: (int)
        Maximum batch size to use.
    """

    def __init__(
        self,
        args: argparse.Namespace,
        cent_obs_space: gym.Space,
        node_obs_space: gym.Space,
        edge_obs_space: gym.Space,
        device=torch.device("cpu"),
        split_batch: bool = False,
        max_batch_size: int = 32,
    ) -> None:
        super(GR_Critic, self).__init__()
        self.args = args
        self.hidden_size = args.hidden_size
        self._use_orthogonal = args.use_orthogonal
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self._use_popart = args.use_popart
        self.split_batch = split_batch
        self.max_batch_size = max_batch_size
        self.tpdv = dict(dtype=torch.float32, device=device)
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][
            self._use_orthogonal
        ]

        cent_obs_shape = get_shape_from_obs_space(cent_obs_space)
        node_obs_shape = get_shape_from_obs_space(node_obs_space)[
            1
        ]  # (num_nodes, num_node_feats)
        edge_dim = get_shape_from_obs_space(edge_obs_space)[0]  # (edge_dim,)

        # TODO modify output of GNN to be some kind of global aggregation
        self.gnn_base = GNNBase(args, node_obs_shape, edge_dim, args.critic_graph_aggr)
        gnn_out_dim = self.gnn_base.out_dim
        # if node aggregation, then concatenate aggregated node features for all agents
        # otherwise, the aggregation is done for the whole graph
        if args.critic_graph_aggr == "node":
            gnn_out_dim *= args.num_agents
        mlp_base_in_dim = gnn_out_dim
        if self.args.use_cent_obs:
            mlp_base_in_dim += cent_obs_shape[0]

        self.base = MLPBase(args, cent_obs_shape, override_obs_dim=mlp_base_in_dim)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(
                self.hidden_size,
                self.hidden_size,
                self._recurrent_N,
                self._use_orthogonal,
            )

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        if self._use_popart:
            self.v_out = init_(PopArt(self.hidden_size, 1, device=device))
        else:
            self.v_out = init_(nn.Linear(self.hidden_size, 1))

        self.to(device)

    def forward(
        self, cent_obs, node_obs, adj, agent_id, rnn_states, masks
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute actions from the given inputs.
        cent_obs: (np.ndarray / torch.Tensor)
            Observation inputs into network.
        node_obs (np.ndarray):
            Local agent graph node features to the actor.
        adj (np.ndarray):
            Adjacency matrix for the graph.
        agent_id (np.ndarray / torch.Tensor)
            The agent id to which the observation belongs to
        rnn_states: (np.ndarray / torch.Tensor)
            If RNN network, hidden states for RNN.
        masks: (np.ndarray / torch.Tensor)
            Mask tensor denoting if RNN states
            should be reinitialized to zeros.

        :return values: (torch.Tensor) value function predictions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        cent_obs = check(cent_obs).to(**self.tpdv)
        node_obs = check(node_obs).to(**self.tpdv)
        adj = check(adj).to(**self.tpdv)
        agent_id = check(agent_id).to(**self.tpdv).long()
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        # if batch size is big, split into smaller batches, forward pass and then concatenate
        if (self.split_batch) and (cent_obs.shape[0] > self.max_batch_size):
            # print(f'Cent obs: {cent_obs.shape[0]}')
            batchGenerator = minibatchGenerator(
                cent_obs, node_obs, adj, agent_id, self.max_batch_size
            )
            critic_features = []
            for batch in batchGenerator:
                obs_batch, node_obs_batch, adj_batch, agent_id_batch = batch
                nbd_feats_batch = self.gnn_base(
                    node_obs_batch, adj_batch, agent_id_batch
                )
                act_feats_batch = torch.cat([obs_batch, nbd_feats_batch], dim=1)
                critic_feats_batch = self.base(act_feats_batch)
                critic_features.append(critic_feats_batch)
            critic_features = torch.cat(critic_features, dim=0)
        else:
            nbd_features = self.gnn_base(
                node_obs, adj, agent_id
            )  # CHECK from where are these agent_ids coming
            if self.args.use_cent_obs:
                critic_features = torch.cat(
                    [cent_obs, nbd_features], dim=1
                )  # NOTE can remove concatenation with cent_obs and just use graph_feats
            else:
                critic_features = nbd_features
            critic_features = self.base(critic_features)  # Cent obs here

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)
        values = self.v_out(critic_features)

        return (values, rnn_states)
