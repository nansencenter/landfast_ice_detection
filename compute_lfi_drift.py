import argparse
from datetime import datetime

import numpy as np
from omegaconf import OmegaConf

from utils import create_destination_domain, read_s1_sar_gdf, get_overlapping_gdf, LoopProcessor, stack_safe_zip_files, stack_nc_files

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process configuration file for landfast ice detection.')
    parser.add_argument('config_file', type=str, help='Path to the configuration file')
    return parser.parse_args()

def read_conf(config_file):
    conf = OmegaConf.load('base.yml')
    conf = OmegaConf.merge(conf, OmegaConf.load(config_file))
    conf.warp_cval = eval(conf.warp_cval)
    period_start_date = datetime.strptime(conf.period_start_date, '%Y-%m-%d')
    stack_function = eval(conf.stack_function)
    return conf, period_start_date, stack_function

def main(config_file):
    conf, period_start_date, stack_function = read_conf(config_file)
    dst_dom, landmask, dst_polygon = create_destination_domain(conf)
    gdf = read_s1_sar_gdf(conf)
    overlapping_gdf = get_overlapping_gdf(conf, gdf, dst_polygon)
    lp  = LoopProcessor(conf, dst_dom, landmask, overlapping_gdf, period_start_date, stack_function)
    lp.loop()

if __name__ == "__main__":
    args = parse_arguments()
    print(f"Configuration file: {args.config_file}")
    main(args.config_file)
