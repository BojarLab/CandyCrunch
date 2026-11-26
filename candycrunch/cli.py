#!/usr/bin/env python3

import argparse
import glob
import os
from candycrunch import __version__

def str_to_bool(value):
    if value.lower() in ('true', '1', 'yes'):
        return True
    if value.lower() in ('false', '0', 'no'):
        return False
    raise argparse.ArgumentTypeError(f"expected True or False, got '{value}'")

def build_parser():
    # The CLI's options, also the settings form of the in-browser page (web.py)
    parser = argparse.ArgumentParser(prog='candycrunch_predict', description='Run CandyCrunch prediction.')
    parser.add_argument('--version', action='version', version=f'CandyCrunch {__version__}')
    parser.add_argument('--spectra_filepath', help='Path(s) to spectra files (Thermo .raw, SCIEX .wiff (with its .wiff.scan), Bruker (timsTOF, Q-TOF, ion traps) and Agilent MassHunter .d folders, Waters .raw folders, .mzML, .mzXML, mzData (.mzdata/.mzdata.xml), .mgf, each also gzipped (.mzML.gz), .cc/.xlsx from extract_spectra) and/or folders of .raw/.wiff/.d/.mzML/.mzXML/.mzdata/.mgf/.cc files; several files are harmonized with wrap_inference_batch', type=str, nargs='+', required=True)
    parser.add_argument('--glycan_class', help='Glycan class, or several measured together in one run (e.g., N O)', type=str, nargs='+', choices=['O', 'N', 'free', 'lipid'], required=True)
    parser.add_argument('--mode', help='Ion mode; the polarity stored in the file overrides it; default: negative', type=str, choices=['negative', 'positive'], required=False)
    parser.add_argument('--max_charge', help='Maximum absolute precursor charge to consider; default: 3', type=int, required=False)
    parser.add_argument('--modification', help='Reducing-end modification: reduced (alditol), reduced_D (NaBD4), unreduced (free reducing end), a label, benzyl (benzyl glycoside), glycolate or lactate (O-glycans released oxidatively, on the glycolic acid of their Ser or the lactic acid of their Thr), or custom: a label of --mass_tag Da; several for glycans of several reducing ends side by side (e.g., glycolate lactate), the first being the run\'s own; default: reduced', type=str, nargs='+', choices=['reduced', 'reduced_D', 'unreduced', '2AA', '2AB', 'procainamide', 'PA', 'APTS', 'aminoxyTMT', 'GirardP', 'benzyl', 'glycolate', 'lactate', 'custom'], required=False)
    parser.add_argument('--mass_tag', help='Mass in Da of a --modification custom label', type=float, required=False)
    parser.add_argument('--sample_prep', help='Sample preparation; default: underivatized', type=str, choices=['underivatized', 'permethylated', 'deuteropermethylated', 'peracetylated'], required=False)
    parser.add_argument('--sialic_derivatization', help='Linkage-specific sialic acid derivatization of underivatized samples (or of samples permethylated after it): EE (ethyl esterification), EEA (ethyl esterification + amidation), ME (methyl esterification), DMA (dimethylamidation), SALSA, GQ (glycoqueuing: aniline on every sialic acid, which tells no linkage; with GirardP), GQ-d5 (d5-aniline on a2-6, d0 on a2-3)', type=str, choices=['EE', 'EEA', 'ME', 'DMA', 'SALSA', 'GQ', 'GQ-d5'], required=False)
    parser.add_argument('--adducts', help='Ions that carry the charge of the precursors in place of protons, e.g., Li+ for lithium salts in the mobile phase; the usual adducts (Na+, K+, and NH4+ in positive mode; acetate, formate, and HCO3- in negative mode) are always considered; default: none (Na+ for permethylated glycans in positive mode)', type=str, nargs='+', required=False)
    parser.add_argument('--run_anion', help='Anion of the mobile phase in negative mode: read before the other anion adducts, and its adduct ions count for abundances; default: all are considered, and with survey scans the anion that neutral glycans show counts for abundances', type=str, choices=['HCO3-', 'Formate', 'Acetate'], required=False)
    parser.add_argument('--reducing_end_variants', help='Other reducing ends the glycans may carry: free (glycans that escaped reduction or labeling, counted for their glycan), ethyl_glycoside (rows of their own), glycosylamine (PA glycans whose label was not reduced, counted for their glycan), glycosylaminitol (reduced N-glycans that were glycosylamines when reduced, counted for their glycan); default: free, plus ethyl_glycoside for lipid, glycosylamine for PA, and glycosylaminitol for reduced N-glycans; give none of them to turn this off', type=str, nargs='*', choices=['free', 'ethyl_glycoside', 'glycosylamine', 'glycosylaminitol'], required=False)
    parser.add_argument('--lc', help='Chromatography: PGC, C18, other (e.g., HILIC), or infusion (direct infusion, no LC; also taken for spectra without retention times); default: PGC', type=str, choices=['PGC', 'C18', 'other', 'infusion'], required=False)
    parser.add_argument('--trap', help='Mass analyzer of the MS2 spectra (the survey scan analyzer is read from the file); the instrument stored in the file overrides it; linear (linear ion trap), orbitrap, amazon (Bruker amaZon), or other (e.g., TOF); default: linear', type=str, choices=['linear', 'orbitrap', 'amazon', 'other'], required=False)
    parser.add_argument('--rt_min', help='Minimum retention time in minutes; default: 0 (runs longer than 20 min then skip their first 2 min)', type=float, required=False)
    parser.add_argument('--rt_max', help='Maximum retention time in minutes; default: 0 (all; in runs longer than 40 min, rows in the last 10%% need a peak of their own over a low background)', type=float, required=False)
    parser.add_argument('--rt_diff', help='Maximum retention time difference in minutes within one peak; default: 1.0', type=float, required=False)
    parser.add_argument('--spectra', help='Whether to output representative spectra; default: False', type=str_to_bool, required=False)
    parser.add_argument('--get_missing', help='Whether to output peaks without prediction; default: False', type=str_to_bool, required=False)
    parser.add_argument('--ppm_thresh', help='Mass tolerance in ppm for peak grouping, composition matching, and ppm error filtering (precursors), and for fragments of ion-trap MS2; default: 300', type=float, required=False)
    parser.add_argument('--pred_thresh', help='Minimum model probability of a structure among the isomers of its composition; default: 0.01', type=float, required=False)
    parser.add_argument('--crumbs_thresh', help='Fragment score (CandyCrumbs annotation score) a peak has to exceed to be kept; 2 lower for files with orbitrap or TOF survey scans; default: 3', type=float, required=False)
    parser.add_argument('--extra_thresh', help='Model probability from which structures of another glycan class are allowed; default: 0.2', type=float, required=False)
    parser.add_argument('--frag_num', help='Number of top fragments to report per spectrum; default: 100', type=int, required=False)
    parser.add_argument('--filter_out', help='Composition elements to filter out/ignore, space-separated; the list replaces the default (HexA), and giving none filters nothing', type=str, nargs='*', required=False)
    parser.add_argument('--supplement', help='Whether to impute biosynthetic intermediates from biosynthetic networks; default: True', type=str_to_bool, required=False)
    parser.add_argument('--experimental', help='Whether to search databases for structures of unexplained peaks; default: True', type=str_to_bool, required=False)
    parser.add_argument('--taxonomy_level', help='Taxonomic level to restrict database searches to; default: Class', type=str, choices=['Species', 'Genus', 'Family', 'Order', 'Class', 'Phylum', 'Kingdom', 'Domain'], required=False)
    parser.add_argument('--taxonomy_filter', help='Taxon at --taxonomy_level to restrict database searches to (e.g., Mammalia for Class, Homo_sapiens for Species); default: Mammalia', type=str, required=False)
    parser.add_argument('--glycan_database', help='Your own glycan database: .txt, .csv, .tsv, or .xlsx files of glycans, one per line or row, and/or glycans themselves, space-separated: structures (IUPAC-condensed, WURCS, GlycoCT, GlyTouCan IDs) and compositions (e.g., H5N4F1 or Hex5HexNAc4dHex1); also finds structures the model does not know, by their fragments; default: none (glycowork\'s database)', type=str, nargs='+', required=False)
    parser.add_argument('--database_mode', help='only: --glycan_database in place of glycowork\'s database, so that only its glycans are reported; add: beside it; default: only', type=str, choices=['only', 'add'], required=False)
    parser.add_argument('--intra_cat_thresh', help='Several files only: minutes the RT of a structure can differ from the mean of its group; default: 1.0', type=float, default=1.0)
    parser.add_argument('--top_n_isomers', help='Several files only: number of isomer groups to retain per composition in each file; default: 5', type=int, default=5)
    parser.add_argument('--n_jobs', help='Number of processes: files processed in parallel, and fragment scoring and supporting peaks within a file; default: 1', type=int, default=1)
    parser.add_argument('--plot_glycans', help='Whether to save the output as .xlsx with SNFG glycan images for all top1 predictions; default: False', type=str_to_bool, required=False)
    parser.add_argument('--composition_format', help="How the output tables write compositions: dict ({'Hex': 5, 'HexNAc': 4, 'dHex': 1}), names (Hex5HexNAc4dHex1), or letters (H5N4F1; A Neu5Ac, G Neu5Gc, S sulfate); default: dict", type=str, choices=['dict', 'names', 'letters'], default='dict')
    parser.add_argument('--output', help='Output file path ending in .csv or .xlsx', type=str, required=True)
    return parser

def main(argv=None):
    # argv: the options as a list (default: the command line)
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.output.endswith(('.csv', '.xlsx')):
        parser.error('--output has to end with .csv or .xlsx')
    if not os.path.isdir(os.path.dirname(os.path.abspath(args.output))):
        parser.error(f'the folder of --output {args.output} does not exist')
    # A table open in Excel or a folder without write access is found now, not when the finished results are written
    existed = os.path.exists(args.output)
    try:
        open(args.output, 'a').close()
    except OSError as e:
        parser.error(f'--output {args.output} cannot be written ({e.strerror}); close it if another program has it open, or choose another path')
    if not existed:
        os.remove(args.output)
    # Windows shells pass wildcards (data\*.mzML) on as typed; they are expanded here, as other shells do
    args.spectra_filepath = [q for p in args.spectra_filepath for q in (sorted(glob.glob(p)) or [p] if any(c in p for c in '*?[') and not os.path.exists(p) else [p])]
    # A .d or Waters .raw folder is one run (a shell completes it with a trailing separator); any other folder gives its runs, but not the ._ files of
    # metadata macOS writes beside every file it copies to a drive or share
    filepaths = [f for p in map(os.path.normpath, args.spectra_filepath) for f in ([os.path.join(p, x) for x in sorted(os.listdir(p)) if x.lower().endswith(('.raw', '.wiff', '.d', '.mzml', '.mzxml', '.mzdata', '.mzdata.xml', '.mgf', '.cc', '.mzml.gz', '.mzxml.gz', '.mgf.gz')) and not x.startswith('._')]
                                                                              if os.path.isdir(p) and not p.lower().endswith(('.d', '.raw')) else [p])]
    # A run given in several files is read once: from its .cc, else from its vendor file rather than a copy converted from it
    from candycrunch.utils import format_composition, one_file_per_run
    filepaths = one_file_per_run(filepaths)
    if not filepaths:
        parser.error('no .raw/.wiff/.d/.mzML/.mzXML/.mzdata/.mgf/.cc files (or gzipped .mzML/.mzXML/.mgf) found in --spectra_filepath')
    missing = [f for f in filepaths if not os.path.exists(f)]
    if missing:
        parser.error(f"not found: {', '.join(missing)}")
    # One reducing end is that end, several side by side a list (the first being the run's own, which --mass_tag belongs to)
    ends = args.modification or []
    if (args.mass_tag is not None) != (ends[:1] == ['custom']):
        parser.error('--mass_tag and --modification custom go together (glycans with a free reducing end: --modification unreduced)')
    if args.max_charge == 0:
        parser.error('--max_charge has to be at least 1 (the ion mode is set by --mode)')
    # Each --glycan_database value is a file of glycans or a glycan, read now, so that a glycan it cannot read ends the run before it starts
    if args.glycan_database:
        from candycrunch.utils import read_glycan_database
        try:
            args.glycan_database = [g for v in args.glycan_database for g in read_glycan_database(v if os.path.isfile(v) else [v])]
        except ValueError as e:
            parser.error(f'--glycan_database: {e}')
    # wrap_inference takes a free reducing end as a custom label without mass, as the app does
    ends = ['custom' if m == 'unreduced' else m for m in ends]
    args.modification = (ends[0] if len(ends) == 1 else ends) if ends else None
    args_dict = {k:v for k, v in vars(args).items() if v is not None and k not in ('spectra_filepath', 'output', 'mode', 'filter_out', 'plot_glycans', 'composition_format', 'intra_cat_thresh', 'top_n_isomers', 'n_jobs')}
    # wrap_inference takes the ion mode from the sign of max_charge
    if args.mode or args.max_charge:
        args_dict['max_charge'] = abs(args.max_charge or 3) * (1 if args.mode == 'positive' else -1)
    if args.filter_out is not None:
        args_dict['filter_out'] = set(args.filter_out)
    if args.reducing_end_variants is not None:
        args_dict['reducing_end_variants'] = {k: v for k, v in {'free': 0, 'ethyl glycoside': 'C2H4', 'glycosylamine': 'C5H4N2-O', 'glycosylaminitol': 'NH3-O'}.items() if k.replace(' ', '_') in args.reducing_end_variants}
    # Imported only now: it loads the model, which --help and argument errors do not need, and every process of --n_jobs imports this module again
    from candycrunch.prediction import wrap_inference, wrap_inference_batch
    from candycrunch.references import add_references
    if len(filepaths) == 1:
        tables = [(args.output, wrap_inference(filepaths[0], n_jobs=args.n_jobs, **args_dict))]
    else:
        combined_batch, inference_dfs = wrap_inference_batch(filepaths, intra_cat_thresh=args.intra_cat_thresh, top_n_isomers=args.top_n_isomers, n_jobs=args.n_jobs, **args_dict)
        stem, ext = os.path.splitext(args.output)
        tables = [(args.output, combined_batch)] + [(f'{stem}_{label}{ext}', df_out) for label, df_out in inference_dfs.items()]
    for path, df_out in tables:
        if isinstance(df_out, tuple):
            # (without the MS1 integrations behind each abundance and the terms of each confidence, which only the app shows)
            df_out, spectra_out = df_out[0].drop(columns=['xic', 'confidence_terms'], errors='ignore'), df_out[1]
            # Rounded as in extract_spectra, which keeps every peak dictionary below Excel's 32,767-character cell limit
            df_out['peak_d'] = [{round(float(mz), 4): float(f'{i:.4g}') for mz, i in d.items()} if isinstance(d, dict) else d for d in spectra_out]
            if 'ms3' in df_out.columns:
                df_out['ms3'] = [[(round(p, 4), {round(float(mz), 4): float(f'{i:.4g}') for mz, i in d.items()}) for p, d in x] if isinstance(x, list) else x for x in df_out['ms3']]
        # Per-file tables are indexed by m/z; the combined feature table has a plain row index that is not worth writing
        df_flat = add_references(df_out.reset_index() if df_out.index.name else df_out)
        if 'composition' in df_flat.columns:
            df_flat['composition'] = [format_composition(c, args.composition_format) for c in df_flat['composition']]
        if path.endswith('.csv'):
            df_flat.to_csv(path, index=False)
        if args.plot_glycans and 'top1_pred' in df_flat.columns:
            from glycowork.motif.draw import plot_glycans_excel
            plot_glycans_excel(df_flat, os.path.splitext(path)[0] + '.xlsx', glycan_col_num='top1_pred')
        elif path.endswith('.xlsx'):
            # (openpyxl, a dependency, writes text as text; xlsxwriter, which pandas takes where installed, turns the references into hyperlinks and
            # leaves a cell empty where they are longer than Excel's 255 characters for a link)
            df_flat.to_excel(path, index=False, engine='openpyxl')

if __name__ == '__main__':
    main()
