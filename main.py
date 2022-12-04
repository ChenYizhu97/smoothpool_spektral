import os
import argparse
import tensorflow as tf
from spektral.datasets import TUDataset
from utils import save_data, ratio_to_number, read_seeds
from train_utils import generate_method_description, run_with_randomsplit

parser = argparse.ArgumentParser(description="Trains and evaluates pooling methods...")
parser.add_argument("--batch", default="32", type=int, help="Batch size. 32 by default.",)
parser.add_argument("--lr", default="0.01", type=float, help="Learning rate. 0.01 by default.",)
parser.add_argument("--epochs", default="400", type=int, help="Maximum epoches. 400 by default.",)
parser.add_argument("-k", default="0.5", type=float, help="Ratio of pooling. 0.5 by default.",)
parser.add_argument("--dataset", default="FRANKENSTEIN", help="Dataset for testing. FRANKENSTEIN by default.",
                    choices=["FRANKENSTEIN", "PROTEINS", "AIDS", "ENZYMES", "COX2", "COX2_MD", "Mutagenicity"]
                    )
parser.add_argument("--pool", default="topkpool", help="Pooling method to use. topkpool by default.",
                    choices=["smoothpool", "topkpool", "sagpool", "diffpool"]
                    )
parser.add_argument("--edges", action="store_true", help="Use edge features.")
parser.add_argument("--augment", action="store_true", help="Connectivity augmentation.")
parser.add_argument("-r", default="20", type=int, help="Running times of evaluation. 30 by default.")
parser.add_argument("-p", default="40", type=int, help="Patience of early stopping. 40 by default.")
parser.add_argument("-d", default="", type=str, help="Addtional description to be logged.")
parser.add_argument("-fseeds", default="random_seeds", type=str, help="File that holds fixed seeds.")

args = parser.parse_args()
#set GPU
physical_devices = tf.config.list_physical_devices("GPU")
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
#load spektral dataset
data = TUDataset(args.dataset)
# Ensure GPU reproducibility
#os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
#os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_GPU_ALLOCATOR']='cuda_malloc_async'


#For diffpool, calculate the fixed numbers of pooled nodes.
if args.pool == "diffpool":
    k = ratio_to_number(args.k, data)
else:
    k = args.k

# run the experiments for args.r times and save the metrics
accs=[]
epochs=[]
#load fixed seeds
seeds = read_seeds(args.fseeds)
method_descriptor = generate_method_description(
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
for i_run in range(args.r):
    #acc, epoch = run_with_10fold(i_run, dataset, seed=seeds[i_run])     
    #print(f"Done. The {i_run+1} run. Test acc: {acc}. Epoch run: {epoch}")
    acc, loss, epoch = run_with_randomsplit( 
        data,
        args.pool,
        args.batch,
        args.epochs,
        args.edges,
        args.lr,
        args.p,
        k,
        seed=seeds[i_run],
    )
    print(f"Done. The {i_run+1} run. Test loss: {loss}. Test acc: {acc}. Epoch run: {epoch}")
    accs.append(acc)
    epochs.append(epoch)
    
file_name = f"DATA_{args.dataset}.txt"
save_data(file_name, accs, method_descriptor, epochs)
