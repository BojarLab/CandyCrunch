#!/usr/bin/env python3

import argparse
import os

def str_to_bool(value):
    if value.lower() in ('true', '1', 'yes'):
        return True
    if value.lower() in ('false', '0', 'no'):
        return False
    raise argparse.ArgumentTypeError(f"expected True or False, got '{value}'")

def main():
    parser = argparse.ArgumentParser(description='Run CandyCrunch prediction.')
    parser.add_argument('--spectra_filepath', help='Path(s) to spectra files (Thermo .raw, Bruker .d folders (timsTOF, ion traps), .mzML, .mzXML, .mgf, .cc/.xlsx from extract_spectra) and/or folders of .raw/.d/.mzML/.mzXML/.mgf/.cc files; several files are harmonized with wrap_inference_batch', type=str, nargs='+', required=True)
    parser.add_argument('--glycan_class', help='Glycan class, or several measured together in one run (e.g. N O)', type=str, nargs='+', choices=['O', 'N', 'free', 'lipid'], required=True)
    parser.add_argument('--mode', help='negative/positive mode', type=str, choices=['negative', 'positive'], required=False)
    parser.add_argument('--max_charge', help='Maximum absolute precursor charge to consider', type=int, required=False)
    parser.add_argument('--modification', help='Reducing-end modification; custom: a label of --mass_tag Da', type=str, choices=['reduced', '2AA', '2AB', 'procainamide', 'PA', 'custom'], required=False)
    parser.add_argument('--mass_tag', help='custom tag mass', type=float, required=False)
    parser.add_argument('--sample_prep', help='Sample preparation', type=str, choices=['underivatized', 'permethylated', 'peracetylated'], required=False)
    parser.add_argument('--sialic_derivatization', help='Linkage-specific sialic acid derivatization of underivatized samples: EE (ethyl esterification), EEA (ethyl esterification + amidation), ME (methyl esterification), DMA (dimethylamidation), SALSA, GQ (glycoqueuing: aniline on every sialic acid, which tells no linkage), GQ-d5 (d5-aniline on a2-6, d0 on a2-3)', type=str, choices=['EE', 'EEA', 'ME', 'DMA', 'SALSA', 'GQ', 'GQ-d5'], required=False)
    parser.add_argument('--reducing_end_variants', help='Other reducing ends the glycans may carry: free (glycans that escaped reduction or labelling, counted for their glycan), ethyl_glycoside (rows of their own); default: free, plus ethyl_glycoside for lipid; give none of them to turn this off', type=str, nargs='*', choices=['free', 'ethyl_glycoside'], required=False)
    parser.add_argument('--lc', help='LC type', type=str, choices=['PGC', 'C18', 'other'], required=False)
    parser.add_argument('--trap', help='Detector type', type=str, choices=['linear', 'orbitrap', 'amazon', 'other'], required=False)
    parser.add_argument('--rt_min', help='Minimum relevant retention time', type=float, required=False)
    parser.add_argument('--rt_max', help='Maximum relevant retention time', type=float, required=False)
    parser.add_argument('--rt_diff', help='Maximum retention time difference within one peak', type=float, required=False)
    parser.add_argument('--spectra', help='Whether to output representative spectra', type=str_to_bool, required=False)
    parser.add_argument('--get_missing', help='Whether to output peaks without prediction', type=str_to_bool, required=False)
    parser.add_argument('--ppm_thresh', help='Mass tolerance in ppm for peak grouping, composition matching and ppm error filtering', type=float, required=False)
    parser.add_argument('--pred_thresh', help='Prediction confidence threshold', type=float, required=False)
    parser.add_argument('--crumbs_thresh', help='CandyCrumbs annotation score a prediction has to exceed to be kept', type=float, required=False)
    parser.add_argument('--extra_thresh', help='Confidence threshold to allow cross-class predictions', type=float, required=False)
    parser.add_argument('--frag_num', help='Number of top fragments to report per spectrum', type=int, required=False)
    parser.add_argument('--filter_out', help='Composition elements to filter out/ignore, space-separated', type=str, nargs='*', required=False)
    parser.add_argument('--supplement', help='Whether to use biosynthetic modeling for zero-shot prediction', type=str_to_bool, required=False)
    parser.add_argument('--experimental', help='Whether to use database searches for zero-shot prediction', type=str_to_bool, required=False)
    parser.add_argument('--taxonomy_level', help='Taxonomic level to restrict database searches to', type=str, choices=['Species', 'Genus', 'Family', 'Order', 'Class', 'Phylum', 'Kingdom', 'Domain'], required=False)
    parser.add_argument('--taxonomy_filter', help='Taxon at taxonomy_level to restrict database searches to', type=str, required=False)
    parser.add_argument('--intra_cat_thresh', help='Several files only: minutes the RT of a structure can differ from the mean of its group', type=float, required=False)
    parser.add_argument('--top_n_isomers', help='Several files only: number of isomer groups to retain per composition in each file; default: 5', type=int, default=5)
    parser.add_argument('--n_jobs', help='Number of processes: files processed in parallel, and fragment scoring within a file; default: 1', type=int, default=1)
    parser.add_argument('--plot_glycans', help='Whether to save the output as .xlsx with SNFG glycan images for all top1 predictions', type=str_to_bool, required=False)
    parser.add_argument('--output', help='Output file path ending in .csv or .xlsx', type=str, required=True)
    args = parser.parse_args()
    if not args.output.endswith(('.csv', '.xlsx')):
        parser.error('--output has to end with .csv or .xlsx')
    if not os.path.isdir(os.path.dirname(os.path.abspath(args.output))):
        parser.error(f'the folder of --output {args.output} does not exist')
    # A timsTOF .d folder is one run (a shell completes it with a trailing separator); any other folder gives its runs
    filepaths = [f for p in map(os.path.normpath, args.spectra_filepath) for f in ([os.path.join(p, x) for x in sorted(os.listdir(p)) if x.lower().endswith(('.raw', '.d', '.mzml', '.mzxml', '.mgf', '.cc'))]
                                                                              if os.path.isdir(p) and not p.lower().endswith('.d') else [p])]
    if not filepaths:
        parser.error('no .raw/.d/.mzML/.mzXML/.mgf/.cc files found in --spectra_filepath')
    missing = [f for f in filepaths if not os.path.exists(f)]
    if missing:
        parser.error(f"not found: {', '.join(missing)}")
    if len(filepaths) > 1 and args.intra_cat_thresh is None:
        parser.error('--intra_cat_thresh is required when processing several files')
    args_dict = {k:v for k, v in vars(args).items() if v is not None and k not in ('spectra_filepath', 'output', 'mode', 'filter_out', 'plot_glycans', 'intra_cat_thresh', 'top_n_isomers', 'n_jobs')}
    # wrap_inference takes the ion mode from the sign of max_charge
    if args.mode or args.max_charge:
        args_dict['max_charge'] = abs(args.max_charge or 3) * (1 if args.mode == 'positive' else -1)
    if args.filter_out is not None:
        args_dict['filter_out'] = set(args.filter_out)
    if args.reducing_end_variants is not None:
        args_dict['reducing_end_variants'] = {k: v for k, v in {'free': 0, 'ethyl glycoside': 'C2H4'}.items() if k.replace(' ', '_') in args.reducing_end_variants}
    # Imported only now: it loads the model, which --help and argument errors do not need, and every process of --n_jobs imports this module again
    from candycrunch.prediction import wrap_inference, wrap_inference_batch
    if len(filepaths) == 1:
        tables = [(args.output, wrap_inference(filepaths[0], n_jobs=args.n_jobs, **args_dict))]
    else:
        combined_batch, inference_dfs = wrap_inference_batch(filepaths, intra_cat_thresh=args.intra_cat_thresh, top_n_isomers=args.top_n_isomers, n_jobs=args.n_jobs, **args_dict)
        stem, ext = os.path.splitext(args.output)
        tables = [(args.output, combined_batch)] + [(f'{stem}_{label}{ext}', df_out) for label, df_out in inference_dfs.items()]
    for path, df_out in tables:
        if isinstance(df_out, tuple):
            df_out, spectra_out = df_out
            # Rounded as in extract_spectra, which keeps every peak dictionary below Excel's 32,767-character cell limit
            df_out['peak_d'] = [{round(float(mz), 4): float(f'{i:.4g}') for mz, i in d.items()} if isinstance(d, dict) else d for d in spectra_out]
            if 'ms3' in df_out.columns:
                df_out['ms3'] = [[(round(p, 4), {round(float(mz), 4): float(f'{i:.4g}') for mz, i in d.items()}) for p, d in x] if isinstance(x, list) else x for x in df_out['ms3']]
        # Per-file tables are indexed by m/z; the combined feature table has a plain row index that is not worth writing
        df_flat = df_out.reset_index() if df_out.index.name else df_out
        if path.endswith('.csv'):
            df_flat.to_csv(path, index=False)
        if args.plot_glycans and 'top1_pred' in df_flat.columns:
            from glycowork.motif.draw import plot_glycans_excel
            plot_glycans_excel(df_flat, os.path.splitext(path)[0] + '.xlsx', glycan_col_num='top1_pred')
        elif path.endswith('.xlsx'):
            df_flat.to_excel(path, index=False)

if __name__ == '__main__':
    main()
