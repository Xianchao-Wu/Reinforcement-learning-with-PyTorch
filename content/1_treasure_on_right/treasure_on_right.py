import time
import numpy as np
import pandas as pd

np.random.seed(2) # 随机种子

N_STATES = 6
ACTIONS = ['left', 'right'] # 一条一维度的线, 动作是left和right，向左一步，以及向右一步
MAX_EPISODES = 13
FRESH_TIME = 0.3 # fresh time for one move
EPSILON = 0.9 # greedy policy
ALPHA = 0.1 # learning rate
GAMMA = 0.9 # discount factor

def update_env(S, episode, step_counter):
    # 更新显示的“环境”，打印到屏幕上：
    env_list = ['-']*(N_STATES-1) + ['T']
    if S == 'terminal':
        interaction = 'Episode %s: total_steps = %s' % (episode+1, step_counter)
        print('\r{}'.format(interaction),end='')
        time.sleep(2)
        print('\r                                ', end='')
    else:
        env_list[S] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)

def get_env_feedback(S,A):
    # 刻画的是，agent如何和环境environment进行交互
    # S = 当前状态；A = 选择好的一个动作
    # given s, take action a, observe r, s'o
    # given S, take action A, observe R and S_ : NOTE
    if A == 'right':
        if S == N_STATES - 2: # 假设一共6个状态，S=4的时候，右边的S=5其实就是终点了
            S_ = 'terminal' # 所以这里的S_ = next state = 终点状态
            R = 1 # 奖励得分为1
        else:
            S_ = S + 1 # 如果当前状态S，不是4，那么就可以执行一步“右移”的操作
            R = 0 # 不过这个时候，奖励为0分（有点不科学。。。）
    else:
        R = 0 # 只要是向左走，都是reward=0，不给分！
        if S == 0:
            # 含义，当前状态S是最左边的一个state了，如果动作还是left，说明撞墙了，下一个动作仍然停留
            # 在"最左边的状态"，
            S_ = S
        else:
            S_ = S - 1 # 如果当前状态S不是最左边的一个状态，那么执行left动作的时候，下个状态是当前
            # 状态左边的那个状态。
    return S_, R # 下一个state，和reward

def build_q_table(n_states, actions):
    table = pd.DataFrame(np.zeros((n_states, len(actions))),columns=actions)
    print(table)
    '''
       left  right
    0   0.0    0.0
    1   0.0    0.0
    2   0.0    0.0
    3   0.0    0.0
    4   0.0    0.0
    5   0.0    0.0
    '''
    return table

def choose_action(state, q_table):
    # 选择一个action的逻辑
    state_actions = q_table.iloc[state,:]
    # > 0.9的时候，随机选择一个action
    # 或者，初始化，全部为0的情况下：
    if (np.random.uniform()>EPSILON) or ((state_actions == 0).all()):
        action_name = np.random.choice(ACTIONS)
    else:
        # <= 0.9的时候，选择一个当前state下，概率最大的那个action:
        action_name = state_actions.idxmax()
    return action_name

def rl():
    q_table = build_q_table(N_STATES, ACTIONS)
    for episode in range(MAX_EPISODES):
        # 每个episode，产生一个完整的从开始状态到一系列动作以及一系列中间状态，最终到终止状态的一个完整的轨迹.
#         print("episode: ", episode)
#         print("q_table: ", q_table)
        step_counter = 0
        S = 0 # 注意，默认都是从S=0，最左边的状态出发，而直线的最右边是“终止状态”:
        is_terminated = False
        update_env(S, episode, step_counter) # 显示初始环境
        while not is_terminated:
            A = choose_action(S, q_table) # 选择一个动作，有对应的“动作选择函数”
            S_, R = get_env_feedback(S, A) # 执行动作，获取下一个状态，以及“即时奖励”
            q_predict = q_table.loc[S,A] # 估计值
            if S_ != 'terminal':
                q_target = R + GAMMA*q_table.iloc[S_,:].max() # NOTE
                # q_target = r + gamma * max_a' Q(s', a')
            else:
                q_target = R
                is_terminated = True
                
            q_table.loc[S,A] += ALPHA*(q_target - q_predict) # NOTE
            # q_predict = Q(s, a)
            #                   学习率 * (真实值 - 估计值)
            S = S_
            
            update_env(S, episode, step_counter+1) # 执行一个action之后，更新环境
            step_counter += 1 # 执行动作的数量+1
        print('-'*10)
        print(f'episode={episode}')
        print('-'*10)
        print(q_table)
    return q_table

if __name__ == "__main__":
    q_table = rl()
    print('\r\nQ-table:\n')
    print(q_table)

