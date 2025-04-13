import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from tqdm import tqdm 

import torch



def get_spikes(dataset: pd.DataFrame, session_id: int, animal_id: int):
    
    group = dataset[(dataset.session == session_id) & (dataset.animal == animal_id)]
    spikes = []

    for trial_id, trial in tqdm(group.groupby(['trial_id'])):
        spikes.append(trial.filter(like='Neuron').values)

    return spikes


def get_reaches(dataset: pd.DataFrame, session_id: int, animal_id: int, behavior: str):
    
    group = dataset[(dataset.session == session_id) & (dataset.animal == animal_id)]
    reaches = []

    for trial_id, trial in tqdm(group.groupby(['trial_id'])):
        try:
            reaches.append(trial[[f'{behavior}_x', f'{behavior}_y']].values)
        except KeyError:
            print("Are you sure about the behavioral measures you want to get?")
            print("Make sure it is the same as in the columns name, e.g. 'hand_pos'")

    return reaches


def vel_to_pos(vel):
    return torch.cumsum(torch.tensor(vel), dim=-2)


def get_maze_conditions(dataset: pd.DataFrame, session_id: int, animal_id: int):
    
    group = dataset[(dataset.session == session_id) & (dataset.animal == animal_id)]
    conds = []

    for trial_id, trial in tqdm(group.groupby(['trial_id'])):
        conds.append(trial.maze_conditions.values[0])

    return conds


def get_conds_target_pos(reach_conds, active_target_pos):
    # The position of the active target that the animal is supposed to reach.
    cond_target_pos = []

    for cond in torch.unique(reach_conds):
        # All trials in a condition have the same specified final active target
        cond_target_pos.append(torch.tensor(active_target_pos)[reach_conds == cond][0])

    return torch.stack(cond_target_pos)


def get_conds_average_reach(reaches: torch.Tensor, reach_conds: torch.Tensor):
    conds = torch.unique(reach_conds)
    
    average_reaches = torch.stack([reaches[reach_conds == int(cond), :, :].mean(axis=0) for cond in conds], dim=0)
    conds_std = torch.stack([reaches[reach_conds == int(cond), :, :].std(axis=0) for cond in conds], dim=0)
        
    return average_reaches, conds_std


def get_correct_incorrect_reaches_in_cond(dataset: pd.DataFrame, cond:int, bhv: str):
    cor_reaches = {'indcs': [], 'seqs': []}
    inc_reaches = {'indcs': [], 'seqs': []}

    for group_id, group in dataset.groupby(['session', 'animal']):
        for trial_index, trial in group.groupby(['trial_id']):

            if all(trial.maze_conditions == cond) and all(trial.result == 'R'):
                if all(trial.correct_reach):
                    cor_reaches['indcs'].append(trial_index)
                    cor_reaches['seqs'].append(torch.tensor(trial[[f'{bhv}_x', f'{bhv}_y']].values))
                    
                elif not all(trial.correct_reach) and all(trial.result == 'R'):
                    inc_reaches['indcs'].append(trial_index)
                    inc_reaches['seqs'].append(torch.tensor(trial[[f'{bhv}_x', f'{bhv}_y']].values))

    return cor_reaches, inc_reaches


def get_event_bins(dataset: pd.DataFrame, session_id: int, animal_id: int) -> pd.DataFrame:
    
    event_cols = [col for col in dataset.columns if col.startswith('Event')]
    event_bins_df = pd.DataFrame(columns=['trial_id']+event_cols)
    group = dataset[(dataset.session == session_id) & (dataset.animal == animal_id)].groupby(['trial_id'])
    
    for trial_id, trial in group:
        row = []
        trial = trial.reset_index()
        for event in event_cols:
            if len(trial[trial[event] == True]) == 1:
                row.append(int(trial[trial[event] == True].index[0]))
            elif len(trial[trial[event] == True]) == 0:
                row.append(np.nan)
            else:
                continue

        row = [int(trial_id)] + row
        event_bins_df.loc[len(event_bins_df)] = np.array(row)
        
    return event_bins_df


def get_trials_len_count(dataset: pd.DataFrame, session_id: int, animal_id: int):
    
    group = dataset[(dataset.session == session_id) & (dataset.animal == animal_id)].groupby(['trial_id'])
    trial_lens = []
    
    for trial_id, trial in group:
        trial_lens.append(len(trial))

    len_counts = {}
    for l in trial_lens: 
        if f'{l} bins' in len_counts:
            len_counts[f'{l} bins'] += 1
        else:
            len_counts[f'{l} bins'] = 1
            
    return len_counts


def plot_event_bins_dist(event_bins: pd.DataFrame):
    
    event_cols = [col for col in event_bins.columns if col.startswith('Event')]
    for event in event_cols:
        plt.hist(event_bins[event], bins='auto', density=False, alpha=0.7, edgecolor='gray', label=f'{event} (mean: {int(np.round(event_bins[event].mean()))})')

    plt.legend(fontsize=8)
    plt.title("bins of events occurance")
    plt.xlabel("event bin")
    plt.ylabel("num of trials")


def plot_rastor(dataset: pd.DataFrame, session_id: int, animal_id: int, trial_id: int, behavior_to_plot='hand_vel'):
    
    trial_data = dataset[(dataset.trial_id == trial_id) & (dataset.session == session_id) & (dataset.animal == animal_id)]
    
    # Select data for neurons
    neuron_columns = [col for col in trial_data.columns if col.startswith('Neuron')]
    neurons = neuron_columns[:]
    
    # Identify event columns
    event_columns = [col for col in dataset.columns if col.startswith('Event')]
    
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot events per time for the selected neurons
    for neuron in neurons:
        events = trial_data.index[trial_data[neuron] == 1].tolist()
        ax.eventplot(events, lineoffsets=neurons.index(neuron), alpha=1.0, linelengths=0.5, color='gray')
    
    # Plot event indications as vertical lines for each event type
    colors = ['red', 'blue', 'green']
    event_labels = []
    event_positions = []
    
    for idx, event_col in enumerate(event_columns):
        event_indices = trial_data.index[trial_data[event_col] == 1].tolist()
        for event_index in event_indices:
            ax.axvline(x=event_index, linestyle='-', color='yellow', alpha=0.8, label=event_col)
            event_positions.append(event_index)
            event_labels.append(event_col)
    
    # Combine event labels for positions with multiple events
    event_dict = {}
    for pos, label in zip(event_positions, event_labels):
        if pos in event_dict:
            event_dict[pos].add(label)
        else:
            event_dict[pos] = {label}
    
    # Sort events to ensure the ticks are in order
    sorted_event_positions = sorted(event_dict.keys())
    sorted_event_labels = [', '.join(event_dict[pos]) for pos in sorted_event_positions]
    
    # Set x-axis ticks with event indications
    ax.set_xticks(sorted_event_positions)
    ax.set_xticklabels(sorted_event_labels, rotation=45, fontsize=10)
    
    # Plot velocity data for y-axis
    ax2 = ax.twinx()
    ax2.plot(trial_data[f'{behavior_to_plot}_x'], color='black', label=f'{behavior_to_plot}_x')
    ax2.plot(trial_data[f'{behavior_to_plot}_y'], color='red', label=f'{behavior_to_plot}_y')
    
    # Set y-axis label for velocity
    ax2.set_ylabel(f'\n{behavior_to_plot}', fontsize=12)
    
    ax.set_ylabel('neuron\n', fontsize=12)
    ax.set_title(f'Trial {trial_id}\n', fontsize=14)
    plt.legend()
    
    plt.show()
    
    
def plot_cond_avg_reaches(reaches, reach_conds):
    fig = plt.figure(figsize=(6, 6))
    fig.suptitle(f'condition-averaged reaches\n({torch.unique(reach_conds).shape[0]} unique maze conditions)')
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.axis('off')

    for cond in torch.unique(reach_conds): 
        avg_reach = torch.mean(reaches[reach_conds == cond], axis=0)
        traj = torch.cumsum(avg_reach, dim=0)

        hit_point = traj[-1, :]
        reach_angle = np.arctan2(hit_point[0], hit_point[1])
        
        ax.plot(traj[:, 0], traj[:, 1], linewidth=1.0, color=plt.cm.hsv(reach_angle / (2*np.pi) + 0.5), alpha=0.8)
        
        
def plot_single_reaches(reaches, target_pos, n_trials_to_plot):

    trial_plt_dx = torch.randperm(reaches.shape[0])[:n_trials_to_plot]

    fig = plt.figure(figsize=(6, 6))
    fig.suptitle(f'{trial_plt_dx.shape[0]} single reaches')
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.axis('off')
    
    for i in trial_plt_dx:
        traj = torch.cumsum(reaches[i], dim=0)
        hit_point = traj[-1, :]
        reach_angle = np.arctan2(hit_point[0], hit_point[1])
        reach_color = plt.cm.hsv(reach_angle / (2 * np.pi) + 0.5)
        ax.plot(traj[:, 0], traj[:, 1], linewidth=1.0, alpha=0.6, color=reach_color)
            
        fig.tight_layout()
        
        
def plot_reaches_in_cond(dataset: pd.DataFrame, avg_reaches:torch.Tensor, cond_to_plot:int, behavior_to_plot='hand_vel'):

    correct_reaches, incorrect_reaches = get_correct_incorrect_reaches_in_cond(dataset, cond=cond_to_plot, bhv=behavior_to_plot)

    correct_trajectories = correct_reaches['seqs']
    incorrect_trajectories = incorrect_reaches['seqs']

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.set_title(f'\nmaze condition {cond_to_plot} \n\n')
    ax.axis('off')
    avg_reach = np.cumsum(avg_reaches[cond_to_plot], axis=0)
    ax.plot(avg_reach[:, 0], avg_reach[:, 1], linewidth=1.5, color='navy', alpha=0.8, zorder=3, label='average reach')
    
    for ri, traj in enumerate(correct_trajectories):
        traj = np.cumsum(traj, axis=0)
        ax.plot(traj[:, 0], traj[:, 1], linewidth=1.0, color='navy', alpha=0.2, zorder=1, label='correct reaches' if ri == 0 else '')

    for ri, traj in enumerate(incorrect_trajectories):
        traj = np.cumsum(traj, axis=0)
        ax.plot(traj[:, 0], traj[:, 1], linewidth=1.0, color='gold', alpha=0.4, zorder=2, label='incorrect reaches' if ri == 0 else '')

    plt.legend(fontsize=8)#, loc='upper left')
    plt.show()
    
    
def plot_unique_target_pos(dataset, session_id=3, animal_id=1):
    reach_conds = torch.tensor(get_maze_conditions(dataset, session_id, animal_id))
    active_target_pos = torch.tensor(dataset.drop_duplicates(subset='trial_id')[['target_pos_x', 'target_pos_y']].values)
    conds_target_pos = get_conds_target_pos(reach_conds, active_target_pos)
    n_unique_targets = torch.unique(active_target_pos).shape[0]
        
    x, y = conds_target_pos[:, 0], conds_target_pos[:, 1]
    angles = torch.rad2deg(torch.atan2(y, x))
    x, y, angles = x, y.numpy(), angles.numpy()
    
    # Scatter plot
    plt.scatter(x, y, c=angles, cmap='hsv', alpha=0.8)  # Using HSV colormap
    plt.colorbar(label="reach angle (degrees)")  # Color legend
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"{n_unique_targets} unique target positions")
    plt.grid(True)
    plt.show()
    
    
def plot_reaches_in_cond(dataset: pd.DataFrame, avg_reaches:torch.Tensor, cond_to_plot:int, behavior_to_plot='hand_vel'):

    correct_reaches, incorrect_reaches = get_correct_incorrect_reaches_in_cond(dataset, cond=cond_to_plot, bhv=behavior_to_plot)

    correct_trajectories = correct_reaches['seqs']
    incorrect_trajectories = incorrect_reaches['seqs']

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.set_title(f'\nmaze condition {cond_to_plot} \n\n')
    ax.axis('off')
    avg_reach = np.cumsum(avg_reaches[cond_to_plot], axis=0)
    ax.plot(avg_reach[:, 0], avg_reach[:, 1], linewidth=1.5, color='navy', alpha=0.8, zorder=3, label='average reach')
    
    for ri, traj in enumerate(correct_trajectories):
        traj = np.cumsum(traj, axis=0)
        ax.plot(traj[:, 0], traj[:, 1], linewidth=1.0, color='navy', alpha=0.2, zorder=1, label='correct reaches' if ri == 0 else '')

    for ri, traj in enumerate(incorrect_trajectories):
        traj = np.cumsum(traj, axis=0)
        ax.plot(traj[:, 0], traj[:, 1], linewidth=1.0, color='gold', alpha=0.4, zorder=2, label='incorrect reaches' if ri == 0 else '')

    plt.legend(fontsize=8)#, loc='upper left')
    plt.show()
    
    
def plot_reaches_in_conds(dataset_aligned, hand_vel, binsize, reach_conds, bins_before, align_at, n_conds_to_plot, behavior_to_plot):
    n_bins = hand_vel.shape[-2]

    for cond_idx in torch.unique(reach_conds)[:n_conds_to_plot]:
        #if cond_idx == 0:
        #    continue
        cond_idx = int(cond_idx)
        correct_reaches, incorrect_reaches = get_correct_incorrect_reaches_in_cond(dataset_aligned, cond=cond_idx, bhv=behavior_to_plot)
        correct_trajectories = correct_reaches['seqs']
        incorrect_trajectories = incorrect_reaches['seqs']

        fig = plt.figure(figsize=(6, 6))
        axx = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        axy = fig.add_axes([1.1, 0.1, 0.8, 0.8])
        axpos = fig.add_axes([2.1, 0.1, 0.8, 0.8])

        for traj_i, traj in enumerate(correct_trajectories):
            axx.plot(torch.arange(0, traj[:, 0].shape[0], 1)*binsize, traj[:, 0], linewidth=1.0, color='navy', alpha=0.08, zorder=2, label='correct reaches' if traj_i == 0 else '')
            axy.plot(torch.arange(0, traj[:, 1].shape[0], 1)*binsize, traj[:, 1], linewidth=1.0, color='navy', alpha=0.08, zorder=2, label='correct reaches' if traj_i == 0 else '')

            pos_traj = torch.cumsum(traj, dim=0)
            axpos.plot(pos_traj[:, 0], pos_traj[:, 1], linewidth=0.8, alpha=0.08, color='navy', label='single reaches' if traj_i == 0 else '')

        for traj_i, traj in enumerate(incorrect_trajectories):
            axx.plot(torch.arange(0, traj[:, 0].shape[0], 1)*binsize, traj[:, 0], linewidth=1.0, color='gold', alpha=0.6, zorder=2, label='incorrect reaches' if traj_i == 0 else '')
            axy.plot(torch.arange(0, traj[:, 1].shape[0], 1)*binsize, traj[:, 1], linewidth=1.0, color='gold', alpha=0.6, zorder=2, label='incorrect reaches' if traj_i == 0 else '')

            pos_traj = torch.cumsum(traj, dim=0)
            axpos.plot(pos_traj[:, 0], pos_traj[:, 1], linewidth=0.8, alpha=0.6, color='gold', label='incorrect reaches' if traj_i == 0 else '')

        #avg_traj = torch.mean(correct_trajectories[reach_conds[correct_reaches['indcs']] == cond_idx], axis=0)
        #avg_traj = avg_reaches[cond_idx]
        avg_traj = torch.mean(hand_vel[reach_conds == cond_idx], axis=0)
        axx.plot(torch.arange(0, avg_traj[:, 0].shape[0], 1)*binsize, avg_traj[:, 0], linewidth=1.5, color='navy', alpha=0.6, zorder=3, label='average reach')
        axy.plot(torch.arange(0, avg_traj[:, 1].shape[0], 1)*binsize, avg_traj[:, 1], linewidth=1.5, color='navy', alpha=0.6, zorder=3, label='average reach')

        avg_pos_traj = torch.cumsum(avg_traj, dim=0)
        axpos.plot(avg_pos_traj[:, 0], avg_pos_traj[:, 1], linewidth=1.0, alpha=0.6, color='navy', label='average reach')

        axx.axvline(x=bins_before*binsize, color='gray', alpha=0.6, linestyle='--')
        axy.axvline(x=bins_before*binsize, color='gray', alpha=0.6, linestyle='--')

        axx.annotate(f"{align_at}",
                    xy=(bins_before*binsize, axx.get_ylim()[1]),
                    xytext=(bins_before * binsize - (n_bins * binsize * 0.1), (axx.get_ylim()[1] * 1.1)),
                    arrowprops=dict(facecolor='black', alpha=0.4, arrowstyle='->'),
                    fontsize=7, alpha=0.6, ha='center')

        axx.set_title('vel x\n')
        axx.legend(fontsize=8)#, loc='upper right')
        axx.set_ylabel(f'hand velocity', fontsize=10)
        axy.set_title('vel y\n')
        axy.legend(fontsize=8)#, loc='lower right')
        axy.set_xlabel('\ntime (ms)', fontsize=10)

        axpos.set_title(f'hand pos\n')

        max_val = max(abs(avg_pos_traj[:, 0]).max() * 1.2, abs(avg_pos_traj[:, 1]).max() * 1.2)
        axpos.set_xlim(-max_val, max_val)
        axpos.set_ylim(-max_val, max_val)

        axpos.axhline(0, color='gray', linewidth=0.6, linestyle='--')
        axpos.axvline(0, color='gray', linewidth=0.6, linestyle='--')

        axpos.axis('off')

        fig.text(1, 0.95, f'maze condition {cond_idx}\n\n\n\n\n', ha='center', va='center', fontsize=12)
        #fig.savefig(f'reach_trejectories/cond_{cond_idx}.png', bbox_inches='tight')