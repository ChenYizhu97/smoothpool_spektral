import os
import argparse
import tensorflow as tf
from smoothpool.classification.utils import save_data, read_seeds, generate_experiment_descriptor
from smoothpool.classification.train import run_classifier, load_data


parser = argparse.ArgumentParser(description="Trains and evaluates pooling methods...")
parser.add_argument("--batch", default="32", type=int, help="Batch size. 32 by default.",)
parser.add_argument("--lr", default="0.005", type=float, help="Learning rate. 0.005 by default.",)
parser.add_argument("--epochs", default="400", type=int, help="Maximum epoches. 400 by default.",)
parser.add_argument("-k", default="0.5", type=float, help="Ratio of pooling. 0.5 by default.",)
parser.add_argument("--dataset", default="FRANKENSTEIN", help="Dataset for testing. FRANKENSTEIN by default.",
                    choices=["FRANKENSTEIN", "PROTEINS", "AIDS", "ENZYMES", "COX2", "COX2_MD", "Mutagenicity", "ogbg-molbace", "ogbg-molbbbp","ogbg-molclintox", "ogbg-molhiv"]
                    )
parser.add_argument("--pool", default="topkpool", help="Pooling method to use. topkpool by default.",
                    choices=["smoothpool", "topkpool", "sagpool", "diffpool"]
                    )
parser.add_argument("--edges", action="store_true", help="Use edge features.")
parser.add_argument("-r", default="20", type=int, help="Running times of evaluation. 20 by default.")
parser.add_argument("-p", default="50", type=int, help="Patience of early stopping. 50 by default.")
parser.add_argument("-d", default="", type=str, help="Addtional description to be logged.")
parser.add_argument("-fseeds", default="random_seeds", type=str, help="File that holds fixed seeds.")

args = parser.parse_args()
#set GPU
physical_devices = tf.config.list_physical_devices("GPU")
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
os.environ['TF_GPU_ALLOCATOR']='cuda_malloc_async'

# run the experiments for args.r times and save the metrics
accs=[]
losses = []
epochs=[]
#load fixed seeds
seeds = read_seeds(args.fseeds)
experiment_descriptor = generate_experiment_descriptor(
    pool=args.pool,
    k=args.k,
    r=args.r,
    batch=args.batch,
    lr=args.lr,
    epochs=args.epochs,
    p=args.p,
    d=args.d,
    edges=args.edges,
    dataset=args.dataset,
)
dataset, split_idx = load_data(args.dataset)
for i_run in range(args.r):
    acc, loss, epoch = run_classifier( 
        dataset,
        args.pool,
        args.batch,
        args.epochs,
        args.edges,
        args.lr,
        args.p,
        args.k,
        split_idx=split_idx,
        seed=seeds[i_run],
    )
    print(f"Done. The {i_run+1} run. Test loss: {loss}. Test acc: {acc}. Epoch run: {epoch}")
    accs.append(acc)
    losses.append(loss)
    epochs.append(epoch)
    results = [accs, losses, epochs] 

file_name = f"DATA_{args.dataset}.txt"
save_data(file_name, results, experiment_descriptor)
