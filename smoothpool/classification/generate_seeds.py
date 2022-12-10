import argparse
from smoothpool.classification.utils import generate_random_seeds

parser = argparse.ArgumentParser(description="Randomly generate fixed seeds...")
parser.add_argument("-n", default=30, type=int, help="Number of seeds.")
parser.add_argument("-f", default="random_seeds", type=str, help="Name of file to save seeds.")

args = parser.parse_args()

generate_random_seeds(args.f, args.n)
