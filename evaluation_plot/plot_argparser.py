import argparse

def opt_parser():
    usage = 'Merge FL training results and plot figure from static freezing results and Gradually Freezing results.'
    parser = argparse.ArgumentParser(description=usage)
    parser.add_argument('--load_static_path', type=str, default='', help='Load Static Freeze results from path')
    parser.add_argument('--load_gf_path', type=str, default='', help='Load Gradually Freeze results from path')
    
    parser.add_argument('--load_group1', type=str, default='', help='Load S=0.5 results from path')
    parser.add_argument('--load_group2', type=str, default='', help='Load S=1 results')
    return parser.parse_args()