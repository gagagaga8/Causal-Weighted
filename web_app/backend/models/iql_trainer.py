"""
IQLTraininger class LossFunctionandTraining 

LossFunction 
1. ExpectileRegressionLoss V 
2. TDRegressionLoss Q 
3. Advantage-Weighted Behavior CloningLoss policy 
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple
import os


class IQLTrainer:
    """IQLTraininger"""
    
    def __init__(
        self,
        agent,
        learning_rate=3e-4,
        gamma=0.99,
        tau=0.005,
        expectile=0.7,
        temperature=3.0,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Args:
            agent: IQLAgentinstance
            learning_rate: Learning rate
            gamma: -fold 
            tau: target Updatecoefficient
            expectile: ExpectileRegressionParameters 0.7-0.9 
            temperature: AWR Parametersβ 0.5-2.0 
            device: Computation 
        """
        self.agent = agent.to(device)
        self.device = device
        
        # Parameters
        self.gamma = gamma
        self.tau = tau
        self.expectile = expectile
        self.temperature = temperature
        
        # Optimizer
        self.q_optimizer = optim.Adam(agent.q_network.parameters(), lr=learning_rate)
        self.v_optimizer = optim.Adam(agent.v_network.parameters(), lr=learning_rate)
        self.policy_optimizer = optim.Adam(agent.policy_network.parameters(), lr=learning_rate)
        
        # Trainingstatistics
        self.train_steps = 0
    
    def expectile_loss(self, diff, expectile=0.7):
        """
        ExpectileRegressionLoss nonpair L2Loss 
        
        L = |τ - 1(diff < 0)| * diff^2
        
        when diff > 0 V < Q Weightasτ
        when diff < 0 V > Q Weightas1-τ
        
        Args:
            diff: Q(s,a) - V(s)
            expectile: τParameters
        
        Returns:
            loss: expectileLoss
        """
        weight = torch.where(diff > 0, expectile, 1 - expectile)
        return (weight * (diff ** 2)).mean()
    
    def update_v_network(self, states, actions):
        """
        UpdateV ExpectileRegression 
        
        target V(s) ≈ E[Q(s,a)] butpairQ > V give Weight
        
        Args:
            states: (batch_size, state_dim)
            actions: (batch_size,)
        
        Returns:
            v_loss: V Loss
        """
        # Computationwhen Vvalue
        v_values = self.agent.get_v_value(states)
        
        # ComputationQvalue notUpdateQ 
        with torch.no_grad():
            q_values = self.agent.get_q_value(states, actions)
        
        # ExpectileRegressionLoss
        diff = q_values - v_values
        v_loss = self.expectile_loss(diff, self.expectile)
        
        # to 
        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()
        
        return v_loss.item()
    
    def update_q_network(self, states, actions, rewards, next_states, dones):
        """
        UpdateQ TDRegression 
        
        target Q(s,a) = r + γ * (1 - done) * V_target(s')
        
        Args:
            states: (batch_size, state_dim)
            actions: (batch_size,)
            rewards: (batch_size,)
            next_states: (batch_size, state_dim)
            dones: (batch_size,)
        
        Returns:
            q_loss: Q Loss
        """
        # Computationwhen Qvalue
        q_values = self.agent.get_q_value(states, actions)
        
        # ComputationTDtarget 
        with torch.no_grad():
            next_v_values = self.agent.get_target_v_value(next_states)
            q_target = rewards + self.gamma * (1 - dones) * next_v_values
        
        # TDLoss MSE 
        q_loss = F.mse_loss(q_values, q_target)
        
        # to 
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()
        
        return q_loss.item()
    
    def update_policy_network(self, states, actions):
        """
        Updatepolicy Advantage-Weighted Behavior Cloning 
        
        target π(a|s) ∝ exp(β * A(s,a)) * π_behavior(a|s)
        
        where  A(s,a) = Q(s,a) - V(s)
        
        Args:
            states: (batch_size, state_dim)
            actions: (batch_size,)
        
        Returns:
            policy_loss: policyLoss
            avg_advantage: Mean value
        """
        # Computation Function
        with torch.no_grad():
            q_values = self.agent.get_q_value(states, actions)
            v_values = self.agent.get_v_value(states)
            advantages = q_values - v_values
            
            # ComputationWeight exp(β * A(s,a))
            weights = torch.exp(self.temperature * advantages)
            # NormalizeWeight stableTraining
            weights = torch.clamp(weights, max=100.0)
        
        # policypair Probability
        action_probs = self.agent.get_action_probs(states)
        log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)).squeeze(1) + 1e-8)
        
        # Advantage-Weighted BCLoss
        policy_loss = -(weights.squeeze(1) * log_probs).mean()
        
        # to 
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        return policy_loss.item(), advantages.mean().item()
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Execute Training UpdateQ V policy 
        
        Args:
            batch: Package states, actions, rewards, next_states, donesdictionary
        
        Returns:
            losses: each Lossvalue
        """
        # Packagebatch
        states = batch['states'].to(self.device)
        actions = batch['actions'].to(self.device)
        rewards = batch['rewards'].to(self.device)
        next_states = batch['next_states'].to(self.device)
        dones = batch['dones'].to(self.device)
        
        # 1. UpdateQ 
        q_loss = self.update_q_network(states, actions, rewards, next_states, dones)
        
        # 2. UpdateV 
        v_loss = self.update_v_network(states, actions)
        
        # 3. Updatepolicy 
        policy_loss, avg_advantage = self.update_policy_network(states, actions)
        
        # 4. Updatetarget V 
        self.agent.update_target_v(self.tau)
        
        self.train_steps += 1
        
        return {
            'q_loss': q_loss,
            'v_loss': v_loss,
            'policy_loss': policy_loss,
            'avg_advantage': avg_advantage
        }
    
    def save_checkpoint(self, save_dir, epoch):
        """SaveTraining """
        os.makedirs(save_dir, exist_ok=True)
        checkpoint_path = os.path.join(save_dir, f'iql_checkpoint_epoch_{epoch}.pt')
        
        torch.save({
            'epoch': epoch,
            'train_steps': self.train_steps,
            'agent_state_dict': self.agent.state_dict(),
            'q_optimizer': self.q_optimizer.state_dict(),
            'v_optimizer': self.v_optimizer.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'hyperparameters': {
                'gamma': self.gamma,
                'tau': self.tau,
                'expectile': self.expectile,
                'temperature': self.temperature
            }
        }, checkpoint_path)
        
        print(f"[OK] alreadySave: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """LoadingTraining """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.agent.load_state_dict(checkpoint['agent_state_dict'])
        self.q_optimizer.load_state_dict(checkpoint['q_optimizer'])
        self.v_optimizer.load_state_dict(checkpoint['v_optimizer'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        self.train_steps = checkpoint['train_steps']
        
        print(f"[OK] alreadyLoading: {checkpoint_path}")
        print(f"  - Epoch: {checkpoint['epoch']}")
        print(f"  - Train steps: {self.train_steps}")


# toModule
import torch.nn.functional as F


def test_trainer():
    """TestTraininger"""
    print("TestIQLTraininger...")
    
    # IQL can 
    import sys
    sys.path.append(os.path.dirname(__file__))
    from iql_networks import IQLAgent
    
    # Create can andTraininger
    state_dim = 10
    action_dim = 2
    agent = IQLAgent(state_dim, action_dim, hidden_dims=[64, 64])
    trainer = IQLTrainer(agent, learning_rate=1e-3)
    
    # batch
    batch_size = 32
    batch = {
        'states': torch.randn(batch_size, state_dim),
        'actions': torch.randint(0, action_dim, (batch_size,)),
        'rewards': torch.randn(batch_size, 1),
        'next_states': torch.randn(batch_size, state_dim),
        'dones': torch.randint(0, 2, (batch_size, 1)).float()
    }
    
    # ExecuteTrainingstep
    losses = trainer.train_step(batch)
    
    print(f"[OK] QLoss: {losses['q_loss']:.4f}")
    print(f"[OK] VLoss: {losses['v_loss']:.4f}")
    print(f"[OK] policyLoss: {losses['policy_loss']:.4f}")
    print(f"[OK] Mean : {losses['avg_advantage']:.4f}")
    
    # TestSave/Loading
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer.save_checkpoint(tmpdir, epoch=0)
        trainer.load_checkpoint(os.path.join(tmpdir, 'iql_checkpoint_epoch_0.pt'))
    
    print("\n[OK] TrainingerTestthrough ")


if __name__ == "__main__":
    test_trainer()
