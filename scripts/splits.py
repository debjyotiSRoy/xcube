from fastcore.script import *
from fastai.text.all import *
from xcube.data.external import *
from xcube.imports import *

def lbs_freqs(df):
    lbs_fqs = Counter()
    for labels in df.labels: lbs_fqs.update(labels.split(';'))
    return array(list(lbs_fqs.values()))

@call_parse
def main(
    source_url: Param("Source url", str)="XURLs.MIMIC3",
    data:  Param("Filename of the raw data", str)="mimic3-9k",
    train_splits: Param("Size of training splits", str)="[25, 50, 100, 200, 400, 800, 1600, 3200, 4500, 6400, 9000, 12800, 18000, 25600, 36000, 49354]"
):

    "Splitting raw data for demonstrating the effectiveness of planting attention"
    
    source = untar_xxx(eval(source_url))
    data_file = join_path_file(data, source, ext='.csv')
    df = pd.read_csv(data_file,
                 header=0,
                 usecols=['subject_id', '_id', 'text', 'labels', 'num_targets', 'is_valid', 'split'],
                 dtype={'subject_id': str, '_id': str, 'text': str, 'labels': str, 'num_targets': np.int64, 'is_valid': bool, 'split': str})
    df[['text', 'labels']] = df[['text', 'labels']].astype(str)
    print(f"The full dataset {data} has {len(lbs_freqs(df))} labels.")
    print(f"Avg # of instances per label {np.mean(lbs_freqs(df))}")
    print(f"Max # of instances for a label {np.max(lbs_freqs(df))}")
    print(f"Min # of instances for a label {np.min(lbs_freqs(df))}")
    print(f"Median of instances for a label {np.percentile(lbs_freqs(df), 50)}")
    train_splits = json.loads(train_splits)
    files = [source/('_'.join(data.split('_')[:-1])+'_' + str(o) + '.csv') for o in train_splits]
    print(train_splits)
    print(files)
    for splt,file in zip(train_splits, files):
        df_sample = df[~df['is_valid']].sample(n=splt)
        print(f"{file = }, instances per label = {np.mean(lbs_freqs(df_sample))}")
        df_sample = pd.concat((df_sample, df[df['is_valid']]))
        df_sample.to_csv(file, index=False)
    import IPython; IPython.embed()