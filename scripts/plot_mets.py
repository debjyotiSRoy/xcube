from fastcore.script import *
from fastbook import *
# extra imports <remove later>
import warnings; warnings.filterwarnings(action='ignore')
# end extra imports


splits = [(25, 1.4), (50, 1.7), (100, 2.3), (200, 2.9), (400, 4.2), (800, 5.9), (1600, 8.5), (3200, 13.0), (4500, 16.4), (6400, 20.5), (9000, 25.6), (12800, 32.5), (18000, 42.1), (25600, 54.8), (36000, 70.2), (49354, 89.9)]

# Function to check if a string can be converted to numeric
def is_numeric(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def _transform(log_file):
    df = pd.read_csv(log_file)
    del df['time'], df['epoch']
    df['epoch'] = range(len(df))
    df = df[[df.columns[-1]] + list(df.columns[:-1])]
    df = df[df.applymap(is_numeric).all(axis=1)]
    df['valid_precision_at_k'] = df['valid_precision_at_k'].astype(float)
    return df

def plot_metrics(log_path):
    plot_path = log_path/'plots'
    plot_path.mkdir(exist_ok=True, parents=True)
    for s,f in splits:
        log_file = log_path/f'mimic3-9k_clas_{s}.csv'
        log_file_plant = log_path/f'mimic3-9k_clas_{s}_plant.csv'
        if not (log_file.exists() and log_file_plant.exists()): continue
        df = _transform(log_file)
        df_plant = _transform(log_file_plant)
        fig, ax = plt.subplots()
        ax.plot(df['epoch'], df['valid_precision_at_k'], label=f'{s}', color='red', marker='s', linestyle='--')
        ax.plot(df_plant['epoch'], df_plant['valid_precision_at_k'], label=f'{s}_plant', color='blue', marker='o', linestyle='-')
        std_array = [df['valid_precision_at_k'].std()]*len(df)
        std_array_plant = [df_plant['valid_precision_at_k'].std()]*len(df_plant)
        # Create shaded bands for standard deviations
        plt.fill_between(df['epoch'], df['valid_precision_at_k'] - np.array(std_array), df['valid_precision_at_k'] + np.array(std_array), color='red', alpha=0.2)
        plt.fill_between(df_plant['epoch'], df_plant['valid_precision_at_k'] - np.array(std_array_plant), df_plant['valid_precision_at_k'] + np.array(std_array_plant), color='blue', alpha=0.2)
        # import pdb; pdb.set_trace()
        ax.set_xlabel('epochs')
        ax.set_ylabel('p@k')
        ax.set_title(f'Scatter Plot of epochs vs. p@k: split{s}, freq{f}')
        ax.grid(True)
        ax.legend(loc='lower right')
        fig.savefig(plot_path/f'metrics_{s}.png')
        # break

@call_parse
def main(
    log_path: Param("Path of the logs to plot", str)="." 
):
    log_path = Path(log_path)
    # import pdb; pdb.set_trace()
    plot_metrics(log_path)