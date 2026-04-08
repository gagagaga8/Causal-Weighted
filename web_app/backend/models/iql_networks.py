"""
IQL (Implicit Q-Learning) neural network 
   group Q V policy 

 paper: Kostrikov et al., "Offline Reinforcement Learning with Implicit Q-Learning", ICLR 2022
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MLP(nn.Module):
    """ Module"""
    
    def __init__(self, input_dim, hidden_dims, output_dim, activation='relu', dropout=0.1):
        super(MLP, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'elu':
                layers.append(nn.ELU())
            
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            prev_dim = hidden_dim
        
        # Output 
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class QNetwork(nn.Module):
    """
    Q estimatestatus- value Q(s, a)
    
    Input statusFeature + one-hot 
    Output Qvalue
    """
    
    def __init__(self, state_dim, action_dim, hidden_dims=[256, 256], dropout=0.1):
        super(QNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Q(s,a) = MLP([s, a_onehot])
        self.q_network = MLP(
            input_dim=state_dim + action_dim,
            hidden_dims=hidden_dims,
            output_dim=1,
            activation='relu',
            dropout=dropout
        )
    
    def forward(self, state, action):
        """
        Args:
            state: (batch_size, state_dim) statusFeature
            action: (batch_size,) or (batch_size, 1) Index
        
        Returns:
            q_values: (batch_size, 1) Qvalue
        """
        # actionisCorrect 
        if len(action.shape) == 1:
            action = action.unsqueeze(1)
        
        # will Convertasone-hotencode
        action_onehot = F.one_hot(action.long().squeeze(1), num_classes=self.action_dim).float()
        
        # concatenatestatusand 
        sa = torch.cat([state, action_onehot], dim=1)
        
        # ComputationQvalue
        q_value = self.q_network(sa)
        
        return q_value


class VNetwork(nn.Module):
    """
    V estimatestatus value V(s)
    
    Input statusFeature
    Output Vvalue
    """
    
    def __init__(self, state_dim, hidden_dims=[256, 256], dropout=0.1):
        super(VNetwork, self).__init__()
        
        self.state_dim = state_dim
        
        # V(s) = MLP(s)
        self.v_network = MLP(
            input_dim=state_dim,
            hidden_dims=hidden_dims,
            output_dim=1,
            activation='relu',
            dropout=dropout
        )
    
    def forward(self, state):
        """
        Args:
            state: (batch_size, state_dim) statusFeature
        
        Returns:
            v_values: (batch_size, 1) Vvalue
        """
        v_value = self.v_network(state)
        return v_value


class PolicyNetwork(nn.Module):
    """
    policy policy π(a|s)
    
    Input statusFeature
    Output ProbabilityDistribution
    """
    
    def __init__(self, state_dim, action_dim, hidden_dims=[256, 256], dropout=0.1):
        super(PolicyNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # π(a|s) = softmax(MLP(s))
        self.policy_network = MLP(
            input_dim=state_dim,
            hidden_dims=hidden_dims,
            output_dim=action_dim,
            activation='relu',
            dropout=dropout
        )
    
    def forward(self, state):
        """
        Args:
            state: (batch_size, state_dim) statusFeature
        
        Returns:
            action_probs: (batch_size, action_dim) ProbabilityDistribution
        """
        logits = self.policy_network(state)
        action_probs = F.softmax(logits, dim=-1)
        return action_probs
    
    def get_action(self, state, deterministic=False):
        """
        according topolicySampling 
        
        Args:
            state: (batch_size, state_dim) statusFeature
            deterministic: is argmax 
        
        Returns:
            actions: (batch_size,) Sampling 
            log_probs: (batch_size,) pair Probability
        """
        action_probs = self.forward(state)
        
        if deterministic:
            # policy ProbabilityMaximum 
            actions = torch.argmax(action_probs, dim=-1)
            log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)).squeeze(1) + 1e-8)
        else:
            # Randompolicy byProbabilitySampling
            dist = torch.distributions.Categorical(action_probs)
            actions = dist.sample()
            log_probs = dist.log_prob(actions)
        
        return actions, log_probs


class IQLAgent(nn.Module):
    """
    IQL can Q V policy 
     interfaceuse TrainingandInference
    """
    
    def __init__(self, state_dim, action_dim, hidden_dims=[256, 256], dropout=0.1):
        super(IQLAgent, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Three core networks
        self.q_network = QNetwork(state_dim, action_dim, hidden_dims, dropout)
        self.v_network = VNetwork(state_dim, hidden_dims, dropout)
        self.policy_network = PolicyNetwork(state_dim, action_dim, hidden_dims, dropout)
        
        # target V use stableTraining 
        self.target_v_network = VNetwork(state_dim, hidden_dims, dropout)
        self.target_v_network.load_state_dict(self.v_network.state_dict())
    
    def get_q_value(self, state, action):
        """ Qvalue"""
        return self.q_network(state, action)
    
    def get_v_value(self, state):
        """ Vvalue"""
        return self.v_network(state)
    
    def get_target_v_value(self, state):
        """ target Vvalue use Q Update """
        return self.target_v_network(state)
    
    def get_action_probs(self, state):
        """ ProbabilityDistribution"""
        return self.policy_network(state)
    
    def select_action(self, state, deterministic=True):
        """
         Inference use 
        
        Args:
            state: (batch_size, state_dim) statusFeature
            deterministic: is 
        
        Returns:
            actions: (batch_size,) 
        """
        with torch.no_grad():
            # Method1 usepolicy 
            action_probs = self.get_action_probs(state)
            if deterministic:
                actions = torch.argmax(action_probs, dim=-1)
            else:
                dist = torch.distributions.Categorical(action_probs)
                actions = dist.sample()
            
            # Method2 useQvalue stable use Evaluation 
            # q_values_all = []
            # for a in range(self.action_dim):
            #     action_tensor = torch.full((state.shape[0],), a, dtype=torch.long, device=state.device)
            #     q_val = self.get_q_value(state, action_tensor)
            #     q_values_all.append(q_val)
            # q_values_all = torch.cat(q_values_all, dim=1)
            # actions = torch.argmax(q_values_all, dim=-1)
        
        return actions
    
    def update_target_v(self, tau=0.005):
        """
         Updatetarget V 
        
        Args:
            tau: Updatecoefficient EMA 
        """
        for target_param, param in zip(self.target_v_network.parameters(), 
                                       self.v_network.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    
    def save(self, path):
        """SaveModel"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'v_network': self.v_network.state_dict(),
            'policy_network': self.policy_network.state_dict(),
            'target_v_network': self.target_v_network.state_dict(),
            'state_dim': self.state_dim,
            'action_dim': self.action_dim
        }, path)
    
    def load(self, path):
        """LoadingModel"""
        checkpoint = torch.load(path)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.v_network.load_state_dict(checkpoint['v_network'])
        self.policy_network.load_state_dict(checkpoint['policy_network'])
        self.target_v_network.load_state_dict(checkpoint['target_v_network'])


def test_networks():
    """Test """
    print("TestIQL ...")
    
    # Parameters withRLDataset 
    state_dim = 10  # admission_age, gender, weight, SOFA, immunosuppressant, uo, bun, creat, pot, ph
    action_dim = 2  # 0=notstartRRT, 1=startRRT
    batch_size = 32
    
    # CreateIQL can 
    agent = IQLAgent(state_dim, action_dim, hidden_dims=[128, 128])
    
    # Data
    state = torch.randn(batch_size, state_dim)
    action = torch.randint(0, action_dim, (batch_size,))
    
    # Test to 
    q_value = agent.get_q_value(state, action)
    v_value = agent.get_v_value(state)
    action_probs = agent.get_action_probs(state)
    selected_action = agent.select_action(state)
    
    print(f"[OK] Q Output : {q_value.shape}")
    print(f"[OK] V Output : {v_value.shape}")
    print(f"[OK] policy Output : {action_probs.shape}")
    print(f"[OK] : {selected_action.shape}")
    
    # Outputrange
    print(f"\n Probabilityand: {action_probs.sum(dim=1).mean().item():.4f} ( ≈1.0)")
    print(f"Qvaluerange: [{q_value.min().item():.2f}, {q_value.max().item():.2f}]")
    print(f"Vvaluerange: [{v_value.min().item():.2f}, {v_value.max().item():.2f}]")
    
    print("\n[OK] All Testthrough ")


if __name__ == "__main__":
    test_networks()
