from fastcore.script import *
from fastbook import *
# extra imports <remove later>
import warnings; warnings.filterwarnings(action='ignore')
# end extra imports

# splits_mimic3 = L((25, 1.54), (58,1.96), (136, 2.64), (320, 3.79), (750, 5.60), (1756, 8.84), (4111, 15.23), (9623, 26.73), (22525, 49.27))
splits_mimic3 = L((25, 1.54), (58,1.96), (136, 2.64), (750, 5.60), (1756, 8.84), (4111, 15.23), (9623, 26.73), (22525, 49.27))
p_at_5_mimic3_PLANT = L( (25,0.35), (58,0.39), (136,0.47), (750,0.53), (1756,0.60), (4111,0.67), (9623,0.71), (22525,0.74) )
p_at_5_mimic3_LANT = L( (25,0.28), (58,0.32), (136,0.37), (750,0.41), (1756,0.51), (4111,0.63), (9623,0.67), (22525,0.70) )
p_at_15_mimic3_PLANT = L( (25,0.21), (58,0.24), (136,0.29), (750,0.33), (1756,0.38), (4111,0.44), (9623,0.48), (22525,0.51) )
p_at_15_mimic3_LANT = L( (25,0.19), (58,0.21), (136,0.24), (750,0.25), (1756,0.31), (4111,0.39), (9623,0.42), (22525,0.48) )

# splits_mimic4 = L((25, 1.39), (50, 1.59), (165, 2.41), (424, 3.47), (1090, 5.43), (2802, 9.27), (7203, 17.52), (18513, 38.03), (49579, 97.68))
splits_mimic4 = L((25, 1.39), (50, 1.59), (165, 2.41), (424, 3.47), (1090, 5.43), (7203, 17.52), (18513, 38.03), (49579, 97.68))
p_at_5_mimic4_PLANT = L((25, 0.25), (50, 0.31), (165, 0.36), (424, 0.41), (1090, 0.50), (7203, 0.56), (18513, 0.62), (49579, 0.68))
p_at_5_mimic4_LANT = L((25, 0.21), (50, 0.24), (165, 0.28), (424, 0.30), (1090, 0.43), (7203, 0.47), (18513, 0.53), (49579, 0.60))

def get_best(log_file):
    df = pd.read_csv(log_file)
    df['valid_precision_at_k'] = df['valid_precision_at_k'].astype(float)
    return df['valid_precision_at_k'].max()

def plot_metrics(log_path, splits, dataset_sz, dataset_name="mimic3", lengeom=10):
    plot_path = log_path/'plots'
    plot_path.mkdir(exist_ok=True, parents=True)
    lant, plant = [], []
    print("Used following files for generating plot:")
    for s,f in splits:
        log_file = log_path/f'{dataset_name}_clas_{s}.csv'
        log_file_plant = log_path/f'{dataset_name}_clas_{s}_plant.csv'
        print(log_file, log_file_plant)
        if not (log_file.exists() and log_file_plant.exists()): continue
        # import pdb; pdb.set_trace()
        # lant += get_best(log_file)
        lant += [(dict(p_at_5_mimic3_LANT)[s], dict(p_at_15_mimic3_LANT)[s])]
        # plant += get_best(log_file_plant)
        plant += [(dict(p_at_5_mimic3_PLANT)[s], dict(p_at_15_mimic3_PLANT)[s])]
    lant, plant = L(lant), L(plant)

    # if dataset_name == "mimic3": plant[-1] = 0.5105 # correction made later  
    print(f"{splits.itemgot(0).zipwith(lant.zipwith(plant))=}")

    # import pdb; pdb.set_trace()    
    
    fig, ax = plt.subplots(figsize=(8,4))
    custom_labels = [f"{s}:{f:.1f}" for s,f in splits]

    # Plot lant and plant for p@5   
    ax.plot(range(len(splits)), lant.itemgot(0), label=r'$\mathsf{LAAT (P@5)}$', color='red', marker='s', linestyle='--')
    ax.plot(range(len(splits)), plant.itemgot(0), label=r'$\mathsf{PLANT (P@5)}$', color='blue', marker='o', linestyle='-')
    # Plot lant and plant for p@15   
    ax.plot(range(len(splits)), lant.itemgot(1), label=r'$\mathsf{LAAT (P@15)}$', color='magenta', marker='s', linestyle='--')
    ax.plot(range(len(splits)), plant.itemgot(1), label=r'$\mathsf{PLANT (P@15)}$', color='green', marker='o', linestyle='-')

    def lant_approx(x_point_plant, y_point_plant, lant, idx, dim=6):
        # y_point_plant = plant[x_point_plant]
        print(f"{(x_point_plant, y_point_plant)=}")
        coeffs = np.polyfit(range(len(lant)), lant, dim)
        poly = np.poly1d(coeffs)
        # We want to know which x_value has the corresponding lant value of y_point_plant
        corres_x_point_lant = (poly-y_point_plant).roots[idx].real
        print(f"{corres_x_point_lant = }")
        print(f"All the roots = {(poly-y_point_plant).roots}")
        return y_point_plant, corres_x_point_lant

    # Lets compute an approximation of lant for p@5
    x_point_plant_pat5 = 2
    y_point_plant = plant.itemgot(0)[x_point_plant_pat5]

    y_point_plant_pat5, corres_x_point_lant_pat5 = lant_approx(x_point_plant_pat5, y_point_plant, lant.itemgot(0), idx=2)

    # let's annotate lant for p@5
    #x axis
    ax.annotate(f'', xy=(corres_x_point_lant_pat5-0.03, 0.1), xytext=(corres_x_point_lant_pat5-0.03, y_point_plant_pat5+0.008),
            arrowprops=dict(arrowstyle='-', lw=1, color='blue'))
    # y axis
    ax.annotate(f'', xy=(-0.25, y_point_plant_pat5-0.0), xytext=(corres_x_point_lant_pat5-0.01, y_point_plant_pat5-0.0),
            arrowprops=dict(arrowstyle='-', lw=1, color='blue'))

    print ("*************") 
    # Lets compute an approximation of lant for p@15
    x_point_plant_pat15 = 2.2
    coeffs = np.polyfit(range(len(plant)), plant.itemgot(1), 7)
    apprx_plant = np.poly1d(coeffs)
    y_point_plant = apprx_plant(x_point_plant_pat15)
    y_point_plant_pat15, corres_x_point_lant_pat15 = lant_approx(x_point_plant_pat15, y_point_plant, lant.itemgot(1), idx=2)

    # Let's annotate lant for p@15
    #x axis
    ax.annotate(f'', xy=(corres_x_point_lant_pat15-0.008, 0.1), xytext=(corres_x_point_lant_pat15-0.008, y_point_plant_pat15+0.008),
            arrowprops=dict(arrowstyle='-', lw=1, color='green'))
    # y axis
    ax.annotate(f'', xy=(-0.25, y_point_plant_pat15-0.0), xytext=(corres_x_point_lant_pat15-0.008, y_point_plant_pat15-0.0),
            arrowprops=dict(arrowstyle='-', lw=1, color='green'))

    ax.set_xlim(-0.25, len(splits)-0.7)
    ax.set_xticks(range(len(splits)), custom_labels, rotation=45)
    ax.set_ylim(0.1, max(max(plant))+0.05)
    ax.set_xlabel('# of train examples: mean of instances/label', fontsize=12)
    ax.set_ylabel(r'$\mathsf{P@k}$', fontsize=12)
    ax.set_title(r'$\mathsf{P@k}$ Vs size of train splits in mimic4-full')
    ax.grid(True)
    ax.legend(loc='best')
    legend = plt.legend(fontsize='small')
    # legend.set_bbox_to_anchor((0.85, 1))  # Adjust the (x, y) position of the legend box
    # legend.set_bbox_to_anchor((0.05, 1))  # Adjust the (x, y) position of the legend box

    fig.savefig(plot_path/f'{dataset_name}_PLANTvsLANT.png', bbox_inches='tight')
    
    # Lets compute an approximation of sizes
    sizes = np.geomspace(25, dataset_sz, lengeom).astype(int)[:-1]
    if dataset_name == "mimic3": sizes = np.delete(sizes, np.where(sizes==320)[0])
    if dataset_name == "mimic4_icd10": sizes = np.delete(sizes, np.where(sizes==2802)[0])
    print(f"{sizes=}")

    coeffs = np.polyfit(np.arange(len(sizes)), sizes, 7)
    apprx_sizes = np.poly1d(coeffs)
    print(f"{apprx_sizes(range(len(sizes)))=}")
    corres_splitsz_pat5 = apprx_sizes(corres_x_point_lant_pat5).astype(int)
    print(f"Precision@5: PLANT trained on {apprx_sizes(x_point_plant_pat5)} matches LAAT trained on {corres_splitsz_pat5 = }.")
    corres_splitsz_pat15 = apprx_sizes(corres_x_point_lant_pat15).astype(int)
    # print(f"{corres_splitsz_pat15 = }")
    print(f"Precision@15: PLANT trained on {apprx_sizes(x_point_plant_pat15)} matches LAAT trained on {corres_splitsz_pat15 = }.")

@call_parse
def main(
    log_path: Param("Path of the logs to plot", str)="../tmp" 
):
    log_path = Path(log_path)
    # plot_path.mkdir(exist_ok=True, parents=True)
    plot_metrics(log_path, splits_mimic3, 52723, dataset_name='mimic3')
    # plot_metrics(log_path, splits_mimic4, 122279, dataset_name='mimic4_icd10')