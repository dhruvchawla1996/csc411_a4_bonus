from __future__ import print_function
from collections import defaultdict
from itertools import count
import numpy as np
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions
from torch.autograd import Variable
import matplotlib.pyplot as plt

# Set seeds
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)

plt.switch_backend('agg')

class Environment(object):
    """
    The Tic-Tac-Toe Environment
    """
    # possible ways to win
    win_set = frozenset([(0,1,2), (3,4,5), (6,7,8), # horizontal
                         (0,3,6), (1,4,7), (2,5,8), # vertical
                         (0,4,8), (2,4,6)])         # diagonal
    # statuses
    STATUS_VALID_MOVE = 'valid'
    STATUS_INVALID_MOVE = 'inv'
    STATUS_WIN = 'win'
    STATUS_TIE = 'tie'
    STATUS_LOSE = 'lose'
    STATUS_DONE = 'done'

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset the game to an empty board."""
        self.grid = np.array([0] * 9) # grid
        self.turn = 1                 # whose turn it is
        self.done = False             # whether game is done
        return self.grid

    def render(self):
        """Print what is on the board."""
        map = {0:'.', 1:'x', 2:'o'} # grid label vs how to plot
        print(''.join(map[i] for i in self.grid[0:3]))
        print(''.join(map[i] for i in self.grid[3:6]))
        print(''.join(map[i] for i in self.grid[6:9]))
        print('====')

    def check_win(self):
        """Check if someone has won the game."""
        for pos in self.win_set:
            s = set([self.grid[p] for p in pos])
            if len(s) == 1 and (0 not in s):
                return True
        return False

    def step(self, action):
        """Mark a point on position action."""
        assert type(action) == int and action >= 0 and action < 9
        # done = already finished the game
        if self.done:
            return self.grid, self.STATUS_DONE, self.done
        # action already have something on it
        if self.grid[action] != 0:
            return self.grid, self.STATUS_INVALID_MOVE, self.done
        # play move
        self.grid[action] = self.turn
        if self.turn == 1:
            self.turn = 2
        else:
            self.turn = 1
        # check win
        if self.check_win():
            self.done = True
            return self.grid, self.STATUS_WIN, self.done
        # check tie
        if all([p != 0 for p in self.grid]):
            self.done = True
            return self.grid, self.STATUS_TIE, self.done
        return self.grid, self.STATUS_VALID_MOVE, self.done

    def random_step(self):
        """Choose a random, unoccupied move on the board to play."""
        pos = [i for i in range(9) if self.grid[i] == 0]
        move = random.choice(pos)
        return self.step(move)

    def play_against_random_first(self, action):
        """Play a move, and then have a random agent play the next move."""
        state, status, done = self.step(action)
        if not done and self.turn == 2:
            state, s2, done = self.random_step()
            if done:
                if s2 == self.STATUS_WIN:
                    status = self.STATUS_LOSE
                elif s2 == self.STATUS_TIE:
                    status = self.STATUS_TIE
                else:
                    raise ValueError("???")
        return state, status, done

    def play_against_random_second(self, action):
        """Have a random agent play the next move, then play a move."""
        state, s1, done = self.random_step()
        if done:
            if s1 == self.STATUS_WIN:
                status = self.STATUS_LOSE
            elif s1 == self.STATUS_TIE:
                status = self.STATUS_TIE
            else:
                raise ValueError("???")
        if not done:
            state, status, done = self.step(action)
        return state, status, done

    def play_against_random(self, action, move="first"):
        if move == "first":
            return self.play_against_random_first(action)
        else:
            return self.play_against_random_second(action)

    def play_against_itself(self, policy, action):
        """Play a move, and then play a move."""
        state, status, done = self.step(action)
        if not done and self.turn == 2:
            action, logprob = select_action(policy, state)
            state, s2, done = self.step(action)
            if done:
                if s2 == self.STATUS_WIN:
                    status = self.STATUS_LOSE
                elif s2 == self.STATUS_TIE:
                    status = self.STATUS_TIE
                else:
                    raise ValueError("???")
        return state, status, done

class Policy(nn.Module):
    """
    The Tic-Tac-Toe Policy
    """
    def __init__(self, input_size=27, hidden_size=256, output_size=9):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        softmax = nn.Softmax()
        return softmax(x)

def select_action(policy, state):
    """Samples an action from the policy at the state."""
    state = torch.from_numpy(state).long().unsqueeze(0)
    state = torch.zeros(3,9).scatter_(0,state,1).view(1,27)
    pr = policy(Variable(state))
    m = torch.distributions.Categorical(pr) 
    action = m.sample()
    log_prob = torch.sum(m.log_prob(action))
    return action.data[0], log_prob

def compute_returns(rewards, gamma=1.0):
    """
    Compute returns for each time step, given the rewards
      @param rewards: list of floats, where rewards[t] is the reward
                      obtained at time step t
      @param gamma: the discount factor
      @returns list of floats representing the episode's returns
          G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + ... 

    >>> compute_returns([0,0,0,1], 1.0)
    [1.0, 1.0, 1.0, 1.0]
    >>> compute_returns([0,0,0,1], 0.9)
    [0.7290000000000001, 0.81, 0.9, 1.0]
    >>> compute_returns([0,-0.5,5,0.5,-10], 0.9)
    [-2.5965000000000003, -2.8850000000000002, -2.6500000000000004, -8.5, -10.0]
    """
    G = [0] * len(rewards)

    for i in range(len(rewards)-1, -1, -1):
        if i == len(rewards)-1: G[i] = rewards[i]
        else: G[i] = rewards[i] + gamma * G[i+1]

    return G

def finish_episode(saved_rewards, saved_logprobs, gamma=1.0):
    """Samples an action from the policy at the state."""
    policy_loss = []
    returns = compute_returns(saved_rewards, gamma)
    returns = torch.Tensor(returns)
    # subtract mean and std for faster training
    returns = (returns - returns.mean()) / (returns.std() +
                                            np.finfo(np.float32).eps)
    for log_prob, reward in zip(saved_logprobs, returns):
        policy_loss.append(-log_prob * reward)
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward(retain_graph=True)
    # note: retain_graph=True allows for multiple calls to .backward()
    # in a single step

#TODO: play around with reward values
def get_reward(status):
    """Returns a numeric given an environment status."""
    return {
            Environment.STATUS_VALID_MOVE  : 1,
            Environment.STATUS_INVALID_MOVE: -250,
            Environment.STATUS_WIN         : 500,
            Environment.STATUS_TIE         : -3,
            Environment.STATUS_LOSE        : -3
    }[status]

def choose_random_move():
    """Flip a fair coin and return 'first' or 'second'"""
    return ("first" if random.random() < 0.5 else "second")

def train_against_random(policy, env, gamma=0.75, log_interval=1000):
    """Train policy gradient."""
    optimizer = optim.Adam(policy.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=10000, gamma=0.9)
    running_reward = 0

    episode_axis = []
    return_axis = []

    win_rate_per_episode_first = []
    loss_rate_per_episode_first = []
    tie_rate_per_episode_first = []

    win_rate_per_episode_second = []
    loss_rate_per_episode_second = []
    tie_rate_per_episode_second = []

    for i_episode in count(1):
        saved_rewards = []
        saved_logprobs = []
        state = env.reset()
        done = False
        move = choose_random_move()
        while not done:
            action, logprob = select_action(policy, state)
            state, status, done = env.play_against_random(action, move)
            reward = get_reward(status)
            saved_logprobs.append(logprob)
            saved_rewards.append(reward)

        R = compute_returns(saved_rewards)[0]
        running_reward += R

        finish_episode(saved_rewards, saved_logprobs, gamma)

        if i_episode % log_interval == 0:
            episode_axis.extend([i_episode])
            return_axis.extend([running_reward/log_interval])

            games_won_first, games_lost_first, games_tied_first, invalid_moves_first = play_games_against_random(policy, env, "first")
            games_won_second, games_lost_second, games_tied_second, invalid_moves_second = play_games_against_random(policy, env, "second")

            win_rate_per_episode_first.extend([games_won_first/100.0])
            loss_rate_per_episode_first.extend([games_lost_first/100.0])
            tie_rate_per_episode_first.extend([games_tied_first/100.0])

            win_rate_per_episode_second.extend([games_won_second/100.0])
            loss_rate_per_episode_second.extend([games_lost_second/100.0])
            tie_rate_per_episode_second.extend([games_tied_second/100.0])

            print('Episode {}\tAverage return: {:.2f}\nFirst move:\tGames Won: {}\tGames Lost:{}\tGames Tied:{}\tInvalid Moves:{}\nSecond move:\tGames Won: {}\tGames Lost:{}\tGames Tied:{}\tInvalid Moves:{}'.format(
                i_episode,
                running_reward / log_interval,
                games_won_first, 
                games_lost_first,
                games_tied_first,
                invalid_moves_first, 
                games_won_second, 
                games_lost_second, 
                games_tied_second, 
                invalid_moves_second))
            running_reward = 0

        if i_episode % (log_interval) == 0:
            torch.save(policy.state_dict(),
                       "ttt/policy-%d.pkl" % i_episode)

        if i_episode % 1 == 0: # batch_size
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if i_episode == 50000:
            plt.figure()
            plt.plot(episode_axis, return_axis)
            plt.xlabel("episode #")
            plt.ylabel("average return")
            plt.title("Training curve of Tic-Tac-Toe model")
            plt.savefig("figures/part1_returns.png")

            plt.figure()
            plt.plot(episode_axis, win_rate_per_episode_first , label = "win rate")
            plt.plot(episode_axis, loss_rate_per_episode_first, label = "loss rate")
            plt.plot(episode_axis, tie_rate_per_episode_first, label = "tie rate")
            plt.xlabel("episode #")
            plt.ylabel("win/loss/tie rates")
            plt.title("Evolution of Win/Loss/Tie rates with training - going first")
            plt.legend()
            plt.savefig("figures/part1_wlt_first.png")

            plt.figure()
            plt.plot(episode_axis, win_rate_per_episode_second , label = "win rate")
            plt.plot(episode_axis, loss_rate_per_episode_second, label = "loss rate")
            plt.plot(episode_axis, tie_rate_per_episode_second, label = "tie rate")
            plt.xlabel("episode #")
            plt.ylabel("win/loss/tie rates")
            plt.title("Evolution of Win/Loss/Tie rates with training - going second")
            plt.legend()
            plt.savefig("figures/part1_wlt_second.png")

            return

def train_against_itself(policy, env, gamma=0.75, log_interval=1000):
    """Train policy gradient."""
    optimizer = optim.Adam(policy.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=10000, gamma=0.9)
    running_reward = 0

    episode_axis = []
    return_axis = []

    win_rate_per_episode_first = []
    loss_rate_per_episode_first = []
    tie_rate_per_episode_first = []

    win_rate_per_episode_second = []
    loss_rate_per_episode_second = []
    tie_rate_per_episode_second = []

    for i_episode in count(1):
        saved_rewards = []
        saved_logprobs = []
        state = env.reset()
        done = False
        while not done:
            action, logprob = select_action(policy, state)
            state, status, done = env.play_against_itself(policy, action)
            reward = get_reward(status)
            saved_logprobs.append(logprob)
            saved_rewards.append(reward)

        R = compute_returns(saved_rewards)[0]
        running_reward += R

        finish_episode(saved_rewards, saved_logprobs, gamma)

        if i_episode % log_interval == 0:
            episode_axis.extend([i_episode])
            return_axis.extend([running_reward/log_interval])

            games_won_first, games_lost_first, games_tied_first, invalid_moves_first = play_games_against_random(policy, env, "first")
            games_won_second, games_lost_second, games_tied_second, invalid_moves_second = play_games_against_random(policy, env, "second")

            win_rate_per_episode_first.extend([games_won_first/100.0])
            loss_rate_per_episode_first.extend([games_lost_first/100.0])
            tie_rate_per_episode_first.extend([games_tied_first/100.0])

            win_rate_per_episode_second.extend([games_won_second/100.0])
            loss_rate_per_episode_second.extend([games_lost_second/100.0])
            tie_rate_per_episode_second.extend([games_tied_second/100.0])
            
            print('Episode {}\tAverage return: {:.2f}\nFirst move:\tGames Won: {}\tGames Lost:{}\tGames Tied:{}\tInvalid Moves:{}\nSecond move:\tGames Won: {}\tGames Lost:{}\tGames Tied:{}\tInvalid Moves:{}'.format(
                i_episode,
                running_reward / log_interval,
                games_won_first, 
                games_lost_first,
                games_tied_first,
                invalid_moves_first, 
                games_won_second, 
                games_lost_second, 
                games_tied_second, 
                invalid_moves_second))
            running_reward = 0

        if i_episode % (log_interval) == 0:
            torch.save(policy.state_dict(),
                       "ttt/policy-%d.pkl" % i_episode)

        if i_episode % 1 == 0: # batch_size
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if i_episode == 50000:
            plt.figure()
            plt.plot(episode_axis, return_axis)
            plt.xlabel("episode #")
            plt.ylabel("average return")
            plt.title("Training curve of Tic-Tac-Toe model")
            plt.savefig("figures/part2_returns.png")

            plt.figure()
            plt.plot(episode_axis, win_rate_per_episode_first , label = "win rate")
            plt.plot(episode_axis, loss_rate_per_episode_first, label = "loss rate")
            plt.plot(episode_axis, tie_rate_per_episode_first, label = "tie rate")
            plt.xlabel("episode #")
            plt.ylabel("win/loss/tie rates")
            plt.title("Evolution of Win/Loss/Tie rates with training - going first")
            plt.legend()
            plt.savefig("figures/part2_wlt_first.png")

            plt.figure()
            plt.plot(episode_axis, win_rate_per_episode_second , label = "win rate")
            plt.plot(episode_axis, loss_rate_per_episode_second, label = "loss rate")
            plt.plot(episode_axis, tie_rate_per_episode_second, label = "tie rate")
            plt.xlabel("episode #")
            plt.ylabel("win/loss/tie rates")
            plt.title("Evolution of Win/Loss/Tie rates with training - going second")
            plt.legend()
            plt.savefig("figures/part2_wlt_second.png")

            return

def first_move_distr(policy, env):
    """Display the distribution of first moves."""
    state = env.reset()
    state = torch.from_numpy(state).long().unsqueeze(0)
    state = torch.zeros(3,9).scatter_(0,state,1).view(1,27)
    pr = policy(Variable(state))
    return pr.data


def load_weights(policy, episode):
    """Load saved weights"""
    weights = torch.load("ttt/policy-%d.pkl" % episode)
    policy.load_state_dict(weights)

def play_games_against_random(policy, env, move="first", games = 100):
    """Play games against random and return number of games won, lost or tied"""
    games_won, games_lost, games_tied, invalid_moves = 0, 0, 0, 0

    for i in range(games):
        state = env.reset()
        done = False
        print("Game: %s"%i)

        while not done:
            action, logprob = select_action(policy, state)
            state, status, done = env.play_against_random(action, move)
            invalid_moves += (1 if status == env.STATUS_INVALID_MOVE else 0)
            env.render()

        if status == env.STATUS_WIN: games_won += 1
        elif status == env.STATUS_LOSE: games_lost += 1
        else: games_tied += 1

    return games_won, games_lost, games_tied, invalid_moves

if __name__ == '__main__':
    import sys
    policy = Policy()
    env = Environment()

    if len(sys.argv) == 1:
        # `python tictactoe.py` to train the agent
        train_against_random(policy, env)
    else:
        # `python tictactoe.py <ep>` to print the first move distribution
        # using weightt checkpoint at episode int(<ep>)
        ep = int(sys.argv[1])
        load_weights(policy, ep)
        
        train_against_itself(policy, env)
        #play_games_against_random(policy, env, "first", 2)
        #play_games_against_random(policy, env, "second", 3)
