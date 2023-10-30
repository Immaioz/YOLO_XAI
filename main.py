import argparse
from data import Utils
import numpy as np

parser = argparse.ArgumentParser(description="Explainable AI computation using YoloV8")


parser.add_argument('-d', "-D", help='Relative path to the directory containing .pkl files.', type=str, default='dati/trained_model/')
parser.add_argument("-t", "-T", help="Select confidence (0), iou (1) or initial confidence (2).", type=int, default=0)
parser.add_argument("-c", "-C", help="Select the specific class (default All).", default=7, type=int)
parser.add_argument("-f", "-F", help="Print on file", action='store_true', default=False) 
parser.add_argument("-s", "-S", help="Set confidence and iou or confidence only (Single Class)", action='store_true', default=False) 



# Parse the command line arguments
args = parser.parse_args()

instance = Utils(args.d, args.c, args.t)
instance._instance()

if args.c != 7:
    instance.plot_initial()
    instance.plot(args.s)
else:
    instance.plt_tot()


if (args.f):
    instance.save_csv()



