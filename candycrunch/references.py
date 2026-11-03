"""The literature behind a structure: the references glycowork's database holds for it, whether or not the structure has a GlyTouCan ID"""
import re

# Sources in the database whose pages are gone (every link tried gave 404 or no answer on 2026-11-03)
DEAD_SOURCES = ('functionalglycomics.org', 'glycostore.org')
# How a link is named if its address holds an identifier; else by its site
LABELS = (('PMID ', r'pubmed\.ncbi\.nlm\.nih\.gov/(\d+)'), ('', r'/(PMC\d+)'), ('DOI ', r'doi\.org/(.+)'))
# {glycan: {address: reference}} and {topology: glycans with references}, once reference_index has read the database
INDEX, TOPOLOGIES = None, None


def _topology(glycan):
    return re.sub(r'\([^)]*\)', '()', glycan.removesuffix('-ol'))


def _linkages_agree(a, b):
    """whether two glycans of one topology can be the same: each linkage alike, unspecified in either, or overlapping (b1-3/4 and b1-4)"""
    for x, y in zip(re.findall(r'\(([^)]*)\)', a), re.findall(r'\(([^)]*)\)', b)):
        (x_start, _, x_end), (y_start, _, y_end) = x.partition('-'), y.partition('-')
        if any(p != q and '?' not in (p, q) for p, q in ((x_start[:1], y_start[:1]), (x_start[1:], y_start[1:]))):
            return False
        if '?' not in (x_end, y_end) and not set(x_end.split('/')) & set(y_end.split('/')):
            return False
    return True


def reference_index():
    """reads the references of glycowork's df_glycan, once (about a second)"""
    global INDEX, TOPOLOGIES
    if INDEX is None:
        from glycowork.glycan_data.loader import df_glycan
        index, topologies = {}, {}
        # Each column of references runs parallel to the observations it documents
        for glycan, *columns in zip(*(df_glycan[c] for c in ('glycan', 'ref', 'Species', 'Species', 'tissue_ref', 'tissue_sample', 'tissue_species', 'disease_ref',
                                                             'disease_association', 'disease_species'))):
            for kind, urls, values, taxa in zip(('species', 'tissues', 'diseases'), columns[::3], columns[1::3], columns[2::3]):
                for url, value, taxon in zip(urls, values, taxa):
                    if not (isinstance(url, str) and url.startswith('http')) or any(source in url for source in DEAD_SOURCES):
                        continue
                    # (a link to a table of a paper, or the same page over http, is that paper once; some DOIs came with their < and > entities cut to <; and >;)
                    url = url.strip().split('#')[0].replace('<;', '<').replace('>;', '>')
                    site = re.sub(r'^https?://(www\.)?', '', url).rstrip('/')
                    pubmed = re.search(LABELS[0][1], url)
                    label = next((prefix + m[1] for prefix, pattern in LABELS if (m := re.search(pattern, url))), site.split('/')[0])
                    reference = index.setdefault(glycan, {}).setdefault(site, {'url': url, 'label': label if len(label) <= 30 else f'{label[:29]}…',
                                                                               'pubmed': int(pubmed[1]) if pubmed else 0, 'species': set(), 'tissues': set(), 'diseases': set()})
                    reference[kind].add(value)
                    reference['species'].add(taxon)
            if glycan in index:
                topologies.setdefault(_topology(glycan), []).append(glycan)
        INDEX, TOPOLOGIES = index, topologies
    return INDEX


def _merged(glycans, leave_out = ()):
    """the references of these glycans by address, each with the species, tissues, and diseases it reports, as text"""

    def few(values):
        values = sorted({v for v in values if isinstance(v, str) and v}, key = lambda v: (v != 'Homo_sapiens', v))
        return ', '.join(v.replace('_', ' ') for v in values[:3]) + (f' +{len(values) - 3}' if len(values) > 3 else '')

    merged = {}
    for glycan in glycans:
        for site, reference in INDEX.get(glycan, {}).items():
            if site not in leave_out:
                entry = merged.setdefault(site, {**reference, 'species': set(), 'tissues': set(), 'diseases': set()})
                for kind in ('species', 'tissues', 'diseases'):
                    entry[kind] |= reference[kind]
    for entry in merged.values():
        entry['human'] = 'Homo_sapiens' in entry['species']
        entry.update({kind: few(entry[kind]) for kind in ('species', 'tissues', 'diseases')})
    return merged


def glycan_references(structure):
    """the references glycowork's database holds for a structure\n
   | Arguments:
   | :-
   | structure (string): glycan in IUPAC-condensed nomenclature, as an alditol or not (the pipeline writes reduced glycans without -ol; the database, with it)\n
   | Returns:
   | :-
   | Returns the references of this structure and those of the structures it can be if its linkages are specified differently (Gal(b1-3/4)GlcNAc and
   | Gal(b1-4)GlcNAc), as two lists of dictionaries with url, label, species, tissues, and diseases (the last three as text), those reporting
   | human glycans first, then the most recent PubMed entries
   """
    reference_index()
    name = structure.removesuffix('-ol')
    exact = _merged((name, f'{name}-ol'))
    related = _merged([g for g in TOPOLOGIES.get(_topology(name), []) if g.removesuffix('-ol') != name and _linkages_agree(name, g)], leave_out = exact)
    return tuple(sorted(refs.values(), key = lambda r: (not r['human'], -r['pubmed'], r['label'])) for refs in (exact, related))


def add_references(df):
    """adds a references column (links joined by '; ') to a table of results, after its GlyTouCan_ID column; tables without structures stay as they are"""
    if 'top1_pred' not in df.columns:
        return df
    urls = ['; '.join(r['url'] for r in glycan_references(g)[0]) if isinstance(g, str) else '' for g in df['top1_pred']]
    out = df.copy()
    out.insert(out.columns.get_loc('GlyTouCan_ID') + 1 if 'GlyTouCan_ID' in out.columns else len(out.columns), 'references', urls)
    return out
