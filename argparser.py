from argparse import ArgumentParser

def make_args():
    parser = ArgumentParser()

    parser.add_argument('--seed', dest='seed', default=31415, type=int)
    parser.add_argument('--label', dest='label', default="untitled", type=str,
                        help="Brief description of type of run conducted")
    parser.add_argument('--loglevel', dest='log_level', default=4, type=int,
                        help="Level of logging to filter: 3:WARNING, 4:INFO, 5:DEBUG")
    parser.add_argument('--noDblRing', dest='no_dbl_ring', default=0, type=int,
                        help="Remove double ring compounds from data")
    parser.add_argument('-s', '--stratified', dest='stratified', action="store_true",
                        help="Folds use stratified random sampling")
    parser.add_argument('-a', '--atomic_charge', default="", type=str,
                        help="Include atomic charge for featurisation: g for Gasteiger, c for CHelpG")
    parser.add_argument('-g', '--geom', dest='geom', default="", type=str,
                        help="Use gaussian data for geometry (default: use RDKit)")
    
    # training params
    parser.add_argument('--lr', dest='lr', default=1e-4, type=float)
    parser.add_argument('--batch', dest='batch', default=32, type=int)
    parser.add_argument('--epoch', dest='epoch', default=400, type=int,
                        help="Number of epochs to train for")
    parser.add_argument('--dropout', dest='dropout', default=0.0, type=float,
                        help="Probability of dropout for nodes")

    # testing params
    parser.add_argument('--cv', dest='cv_fold', default=9, type=int,
                        help="Number of folds for cross validation")
    parser.add_argument('--withhold', dest='withhold', default=30, type=int,
                        help="Number of data points to withhold for testing")
    parser.add_argument('--ldist', dest="ldist", default=0.33, type=float,
                        help="Lambda value of distance for MAT")
    parser.add_argument('--latt', dest="latt", default=0.33, type=float,
                        help="Lambda value of attention for MAT")

    args = parser.parse_args()
    return args
