from fastcore.script import *
from fastbook import *
# extra imports <remove later>
import warnings; warnings.filterwarnings(action='ignore')
# end extra imports

# splits_mimic3 = L((25, 1.54), (58,1.96), (136, 2.64), (320, 3.79), (750, 5.60), (1756, 8.84), (4111, 15.23), (9623, 26.73), (22525, 49.27))
splits_mimic3 = L((25, 1.54), (58,1.96), (136, 2.64), (750, 5.60), (1756, 8.84), (4111, 15.23), (9623, 26.73), (22525, 49.27))
# splits_mimic4 = L((25, 1.39), (50, 1.59), (165, 2.41), (424, 3.47), (1090, 5.43), (2802, 9.27), (7203, 17.52), (18513, 38.03), (49579, 97.68))
splits_mimic4 = L((25, 1.39), (50, 1.59), (165, 2.41), (424, 3.47), (1090, 5.43), (7203, 17.52), (18513, 38.03), (49579, 97.68))

def get_best(log_file):
    df = pd.read_csv(log_file)
    df['valid_precision_at_k'] = df['valid_precision_at_k'].astype(float)
    return df['valid_precision_at_k'].max()

def plot_metrics(log_path, splits, dataset_sz,dataset_name="mimic3", lengeom=10):
    plot_path = log_path/'plots'
    plot_path.mkdir(exist_ok=True, parents=True)
    lant, plant = L(), L()
    print("Used following files for generating plot:")
    for s,f in splits:
        log_file = log_path/f'{dataset_name}_clas_{s}.csv'
        log_file_plant = log_path/f'{dataset_name}_clas_{s}_plant.csv'
        print(log_file, log_file_plant)
        if not (log_file.exists() and log_file_plant.exists()): continue
        lant += get_best(log_file)
        plant += get_best(log_file_plant)

    if dataset_name == "mimic3": plant[-1] = 0.5105 # correction made later  
    print(splits.itemgot(0).zipwith(lant.zipwith(plant)))
        
    fig, ax = plt.subplots(figsize=(8,4))
    custom_labels = [f"{s}:{f:.1f}" for s,f in splits]
    ax.plot(range(len(splits)), lant, label='LANT', color='red', marker='s', linestyle='--')
    ax.plot(range(len(splits)), plant, label='PLANT', color='blue', marker='o', linestyle='-')

    x_point_plant = 5
    y_point_plant = plant[x_point_plant]
    print((x_point_plant, y_point_plant))

    # import pdb; pdb.set_trace()
    # Lets compute an approximation of lant
    dim = 6
    coeffs = np.polyfit(range(len(lant)), lant, dim)
    poly = np.poly1d(coeffs)
    # We want to know which x_value has the corresponding lant value of y_point_plant
    if dataset_name=="mimic4_icd10":
        idx = 1
    elif dataset_name=="mimic3":
        idx=2
    corres_x_point_lant = (poly-y_point_plant).roots[idx].real
    print(f"{corres_x_point_lant = }")
    print(f"All the roots = {(poly-y_point_plant).roots}")


    # Lets compute an approximation of sizes
    sizes = np.geomspace(25, dataset_sz, lengeom).astype(int)[:-1]
    if dataset_name == "mimic3": sizes = np.delete(sizes, np.where(sizes==320)[0])
    if dataset_name == "mimic4_icd10": sizes = np.delete(sizes, np.where(sizes==2802)[0])
    print(f"{sizes=}")

    coeffs = np.polyfit(np.arange(len(sizes)), sizes, 7)
    apprx_sizes = np.poly1d(coeffs)
    corres_splitsz = apprx_sizes(corres_x_point_lant).astype(int)
    print(f"{apprx_sizes(range(len(sizes)))=}")
    print(f"{corres_splitsz = }")


    # import IPython; IPython.embed()

    #x axis
    ax.annotate(f'', xy=(corres_x_point_lant-0.05, 0.1), xytext=(corres_x_point_lant-0.05, y_point_plant+0.001),
            arrowprops=dict(arrowstyle='-', lw=1, color='blue'))
    # y axis
    ax.annotate(f'', xy=(-0.25, y_point_plant-0.002), xytext=(corres_x_point_lant-0.03, y_point_plant-0.002),
            arrowprops=dict(arrowstyle='-', lw=1, color='blue'))


    ax.set_xlim(-0.25, len(splits)-0.7)
    # ax.set_xscale('log')
    ax.set_xticks(range(len(splits)), custom_labels, rotation=45)
    ax.set_ylim(0.1, 0.50)
    ax.set_xlabel('# of train examples: mean of instances/label', fontsize=12)
    ax.set_ylabel(r'$\mathsf{P}@15$', fontsize=12)
    ax.set_title(r'$\mathsf{P@k}$ Vs size of train splits in mimic4-full')
    ax.grid(True)
    ax.legend(loc='upper right')
    legend = plt.legend()
    legend.set_bbox_to_anchor((0.85, 1))  # Adjust the (x, y) position of the legend box

    fig.savefig(plot_path/f'{dataset_name}_PLANTvsLANT.png', bbox_inches='tight')
        # break

@call_parse
def main(
    log_path: Param("Path of the logs to plot", str)="../tmp" 
):
    log_path = Path(log_path)
    # plot_path.mkdir(exist_ok=True, parents=True)
    plot_metrics(log_path, splits_mimic3, 52723, dataset_name='mimic3')
    plot_metrics(log_path, splits_mimic4, 122279, dataset_name='mimic4_icd10')