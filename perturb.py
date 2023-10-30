import argparse
from pert import Perturbation

parser = argparse.ArgumentParser(description="Perturbation of a set of images using YoloV8 models to store confidence and IoU data.")


parser.add_argument('-d', "-D", help='Relative path to the directory containing images.', type=str, default='utils/dataset/subset/')
parser.add_argument('-m', "-M", help='Relative path to the directory containing models', type=str, default='models/')
parser.add_argument("-l", "-L", help="Select the target layer.", default=-2, type=int)
parser.add_argument("-lc", "-LC", help="Show layer check", action='store_true', default=False) 
parser.add_argument("-p", "-P", help="Select perturbation (Remove, Random, Mean).", default=0, type=int)
parser.add_argument("-v", "-V", help="Select double model (IR and Visible or not)", action='store_true', default=False) 
parser.add_argument("-c", "-C", help="Select the specific class (default All).", default=7, type=int)

args = parser.parse_args()

tasks = ["remove", "random", "mean"]


instance = Perturbation(args.d, args.m, args.l, tasks[args.p], args.v, args.c, args.lc)

if args.lc == True:
    instance.test_layers()

elif args.c != 7:
    for i in range(6):
        instance.run(i)
else:
    instance.run(args.c)

