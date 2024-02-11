#!/usr/bin/env python3

import argparse
from CandyCrunch.prediction import wrap_inference

def main():
    parser = argparse.ArgumentParser(description='Run CandyCrunch prediction.')
    #parser.add_argument('-c', '--config', help='Path to the config file', required=True)
    parser.add_argument('--spectra_filepath', help='Path to the spectra file', type=str, required=True)
    parser.add_argument('--glycan_class', help='Glycan class', type=str, required=True)
    parser.add_argument('--mode', help='negative/positive mode', type=str, required=False)
    parser.add_argument('--modification', help='glycan derivatization', type=str, required=False)
    parser.add_argument('--mass_tag', help='custom tag mass', type=float, required=False)
    parser.add_argument('--lc', help='LC type', type=str, required=False)
    parser.add_argument('--trap', help='Detector type', type=str, required=False)
    parser.add_argument('--rt_min', help='Minimum relevant retention time', type=float, required=False)
    parser.add_argument('--rt_max', help='Maximum relevant retention time', type=float, required=False)
    parser.add_argument('--rt_diff', help='Maximum retention time difference within one peak', type=float, required=False)
    parser.add_argument('--spectra', help='Whether to output representative spectra', type=bool, required=False)
    parser.add_argument('--get_missing', help='Whether to output peaks without prediction', type=bool, required=False)
    parser.add_argument('--mass_tolerance', help='Maximum mass difference within one peak', type=float, required=False)
    parser.add_argument('--filter_out', help='Composition elements to filter out/ignore', type=dict, required=False)
    parser.add_argument('--supplement', help='Whether to use biosynthetic modeling for zero-shot prediction', type=bool, required=False)
    parser.add_argument('--experimental', help='Whether to use database searches for zero-shot prediction', type=bool, required=False)
    parser.add_argument('--taxonomy_class', help='Taxonomic class to restrict database searches to', type=str, required=False)
    parser.add_argument('--plot_glycans', help='Whether to save an output.xlsx file with SNFG glycan images for all top1 predictions', type=bool, required=False)
    parser.add_argument('--output', help='Output CSV file path', type=str, required=True)

    args = parser.parse_args()
    args_dict = {k:v for k, v in vars(args).items() if v is not None and k != "output"}
    df_out = wrap_inference(**args_dict)
    if args.output.endswith('.csv'):
        df_out.to_csv(args.output)
    if args.output.endswith('.xlsx') and not args.plot_glycans:
        df_out.to_excel(args.output)

if __name__ == '__main__':
    main()
