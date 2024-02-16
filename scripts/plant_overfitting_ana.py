from fastcore.script import *
from fastbook import *
# extra imports <remove later>
import warnings; warnings.filterwarnings(action='ignore')
# end extra imports

@call_parse
def main(
    dataset_name: Param("Name of the dataset", str)="mimic4_full",
    log_file_plant: Param("PLANT log file", str)="mimic4_icd10_clas_full_plant_experiment" ,
    log_file_laat: Param("LAAT log file", str) ="mimic4_icd10_clas_full",
    log_path: Param("Path of the logs to plot", str)="../tmp", 
):
    log_path = Path(log_path)
    df_plant = pd.read_csv(join_path_file(log_file_plant, log_path, ext='.csv'))
    df_laat = pd.read_csv(join_path_file(log_file_laat, log_path, ext='.csv'))

    # Convert non-numeric values to NaN
    df_plant[['epoch', 'train_loss', 'train_precision_at_k', 'valid_loss', 'valid_precision_at_k', ]] = df_plant[[
    'epoch', 'train_loss', 'train_precision_at_k', 'valid_loss', 'valid_precision_at_k', ]].apply(pd.to_numeric, errors='coerce')
    df_laat[['epoch', 'train_loss', 'train_precision_at_k', 'valid_loss', 'valid_precision_at_k',  ]] = df_laat[[
    'epoch', 'train_loss', 'train_precision_at_k', 'valid_loss', 'valid_precision_at_k', ]].apply(pd.to_numeric, errors='coerce')

    # Drop rows with NaN values
    df_plant.dropna(inplace=True)
    df_laat.dropna(inplace=True)

    df_plant = df_plant.reset_index(drop=True)
    df_laat = df_laat.reset_index(drop=True)
    
    df_plant['epoch'] = range(len(df_plant))
    df_laat['epoch'] = range(len(df_laat))

    fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(16,4))
    plant_pat15 = df_plant['valid_precision_at_k'].values
    plant_train_loss = df_plant['train_loss'].values
    plant_test_loss = df_plant['valid_loss'].values

    laat_pat15 = df_laat['valid_precision_at_k'].values
    laat_train_loss = df_laat['train_loss'].values
    laat_test_loss = df_laat['valid_loss'].values
    
    
    ax1.plot(range(len(plant_train_loss)), plant_train_loss, label=r'$\mathsf{PLANT}$ train', color='olive', linestyle='-')
    ax1.plot(range(len(plant_test_loss)), plant_test_loss, label=r'$\mathsf{PLANT}$ test', color='limegreen', linestyle='-')
    ax1.plot(range(len(laat_train_loss)), laat_train_loss, label=r'$\mathsf{LAAT}$ train', color='magenta', linestyle='--')
    ax1.plot(range(len(laat_test_loss)), laat_test_loss, label=r'$\mathsf{LAAT}$ test', color='red', linestyle='--')
    ax1.set_xlabel('epochs', fontsize=12)
    ax1.set_ylabel('loss', fontsize=12)
    ax1.set_title(r'train/test loss Vs epochs in mimic4-full')
    ax1.grid(True)
    ax1.legend(loc='best')

    ax2.plot(range(len(laat_pat15)), laat_pat15, label=r'$\mathsf{LAAT}$ test', color='red', linestyle='--')
    ax2.plot(range(len(plant_pat15)), plant_pat15, label=r'$\mathsf{PLANT}$ test', color='limegreen', linestyle='-')
    # ax2.set_ylim(min(min(plant), min(laat)) - 0.1, max(max(plant), max(laat))+0.05)
    # ax2.set_ylim(0.35, max(max(plant), max(laat))+0.05)
    ax2.set_xlabel('epochs', fontsize=12)
    ax2.set_ylabel(r'$\mathsf{P@15}$', fontsize=12)
    ax2.set_title(r'test $\mathsf{P@15}$ Vs epochs in mimic4-full')
    ax2.grid(True)
    ax2.legend(loc='best')
    plt.tight_layout()
    plot_path = log_path/'plots'
    fig.savefig(plot_path/f'{dataset_name}_PLANTvsLANT_full.png', bbox_inches='tight')
    # import IPython; IPython.embed()