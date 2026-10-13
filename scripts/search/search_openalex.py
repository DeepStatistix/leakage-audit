"""Systematic search on OpenAlex for the leakage audit (protocol Section 5-6).

Downloads every record matching search strings S1 (general pool) and S2 (decomposition stratum), keeps the raw
records, deduplicates (records in both pools go to S2), shuffles each pool with numpy.random.default_rng(2026),
and writes the screening order plus a search log with date, strings, filters and counts (for PRISMA).

Run:  C:\\Python314\\python.exe scripts/audit/search_openalex.py
Outputs: paper/search/ (raw/*.jsonl.gz, records_all.csv, screening_order_S1.csv, screening_order_S2.csv,
         search_log.md)
"""
import datetime as dt
import gzip
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd
import requests

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUT = os.path.join(ROOT, 'paper', 'search')
API = 'https://api.openalex.org/works'
SEED = 2026

STRINGS = {
    # v2 (7 Oct 2026): all application fields; the v1 domain clause was dropped (archive_domain_limited_v1/)
    'S1': ('("time series" OR "time-series") AND (forecast OR prediction) '
           'AND ("deep learning" OR "machine learning" OR LSTM OR GRU OR transformer OR "neural network" OR hybrid)'),
    'S2': ('(forecast OR prediction) AND ("time series" OR "time-series") '
           'AND (wavelet OR "empirical mode decomposition" OR EEMD OR CEEMDAN OR "variational mode decomposition" '
           'OR VMD OR TQWT OR "singular spectrum analysis" OR "seasonal-trend decomposition") '
           'AND (hybrid OR LSTM OR "neural network" OR "machine learning" OR "deep learning")'),
}
FILTERS = 'publication_year:2018-2025,type:article,language:en,primary_location.source.type:journal'
FIELDS = 'id,doi,title,publication_year,publication_date,primary_location,open_access,authorships,abstract_inverted_index'


def fetch(query):
    """All records for one search string, via cursor paging (200 per page)."""
    params = {'filter': f'title_and_abstract.search:{query},{FILTERS}', 'per_page': 200, 'cursor': '*',
              'select': FIELDS}
    # Without a key, requests share a small free daily budget per IP; a free personal key has its own budget.
    headers = {'Authorization': f'Bearer {os.environ["OPENALEX_API_KEY"]}'} if os.environ.get('OPENALEX_API_KEY') else {}
    records, count = [], None
    while params['cursor']:
        for attempt in range(5):
            try:
                r = requests.get(API, params=params, headers=headers, timeout=120)
            except requests.exceptions.RequestException:  # dropped connection / timeout: wait and retry
                time.sleep(min(120, 5 * 2 ** attempt))
                continue
            if r.status_code == 200:
                break
            if r.status_code == 429 and 'budget' in r.text.lower():
                raise SystemExit('OpenAlex daily budget used up: set OPENALEX_API_KEY or retry after midnight UTC.')
            time.sleep(min(120, 5 * 2 ** attempt))  # transient rate limit: back off
        r.raise_for_status()
        data = r.json()
        count = data['meta']['count']
        records.extend(data['results'])
        params['cursor'] = data['meta'].get('next_cursor')
        if not data['results']:
            break
        time.sleep(1.0)
    return records, count


def abstract(inv):
    if not inv:
        return ''
    pos = {p: w for w, ps in inv.items() for p in ps}
    return ' '.join(pos[i] for i in sorted(pos))


def flatten(rec, pool):
    src = ((rec.get('primary_location') or {}).get('source') or {})
    auth = rec.get('authorships') or []
    return {'openalex_id': rec['id'].rsplit('/', 1)[-1], 'pool': pool,
            'doi': (rec.get('doi') or '').replace('https://doi.org/', '').lower(),
            'title': (rec.get('title') or '').strip(), 'year': rec.get('publication_year'),
            'journal': src.get('display_name'), 'issn_l': src.get('issn_l'), 'publisher': src.get('host_organization_name'),
            'first_author': ((auth[0].get('author') or {}).get('display_name') if auth else None),
            'is_oa': (rec.get('open_access') or {}).get('is_oa'), 'oa_url': (rec.get('open_access') or {}).get('oa_url'),
            'abstract': abstract(rec.get('abstract_inverted_index'))}


def norm_title(t):
    return re.sub(r'[^a-z0-9]+', ' ', (t or '').lower()).strip()


def main():
    os.makedirs(os.path.join(OUT, 'raw'), exist_ok=True)
    today = dt.date.today().isoformat()
    counts, frames = {}, []
    from_raw = '--from-raw' in sys.argv  # rebuild from archived raw records without re-downloading
    if from_raw:
        today = sys.argv[sys.argv.index('--from-raw') + 1]
        api_counts = json.load(open(os.path.join(OUT, 'raw', f'api_counts_{today}.json')))
    for pool, q in STRINGS.items():
        raw = os.path.join(OUT, 'raw', f'{pool}_openalex_{today}.jsonl.gz')
        if from_raw:
            with gzip.open(raw, 'rt', encoding='utf-8') as f:
                recs = [json.loads(line) for line in f]
            count = api_counts[pool]
        else:
            recs, count = fetch(q)
            with gzip.open(raw, 'wt', encoding='utf-8') as f:
                for r in recs:
                    f.write(json.dumps(r) + '\n')
        counts[pool] = (count, len(recs))
        frames.append(pd.DataFrame([flatten(r, pool) for r in recs]))
        print(f'{pool}: API count {count}, downloaded {len(recs)}', flush=True)
    if not from_raw:
        json.dump({p: counts[p][0] for p in STRINGS}, open(os.path.join(OUT, 'raw', f'api_counts_{today}.json'), 'w'))

    allrec = pd.concat(frames, ignore_index=True)
    allrec.to_csv(os.path.join(OUT, 'records_all.csv'), index=False)
    n_raw = len(allrec)
    # dedupe: S2 wins over S1; then by OpenAlex id, DOI and normalised title
    allrec['pool_rank'] = allrec['pool'].map({'S2': 0, 'S1': 1})
    allrec['tkey'] = allrec['title'].map(norm_title)
    allrec = allrec[allrec['tkey'] != ''].sort_values(['pool_rank', 'year'])
    d = allrec.drop_duplicates('openalex_id')
    n_after_id = len(d)
    d = pd.concat([d[d.doi == ''], d[d.doi != ''].drop_duplicates('doi')]).drop_duplicates('tkey')
    in_both = len(set(frames[0].openalex_id) & set(frames[1].openalex_id))

    for pool in ('S1', 'S2'):
        rng = np.random.default_rng(SEED)  # one generator per pool, so each pool's order is independent of the other
        p = d[d.pool == pool].sort_values('openalex_id').reset_index(drop=True)  # canonical order before shuffle
        p = p.iloc[rng.permutation(len(p))].reset_index(drop=True)
        p.insert(0, 'order_no', np.arange(1, len(p) + 1))
        p.drop(columns=['pool_rank', 'tkey']).to_csv(os.path.join(OUT, f'screening_order_{pool}.csv'), index=False)
        counts[pool + '_unique'] = len(p)

    with open(os.path.join(OUT, 'search_log.md'), 'w', encoding='utf-8') as f:
        f.write(f"""# Search log

**Database:** OpenAlex API ({API}), field `title_and_abstract.search` (stemmed)
**Date run:** {today}
**Filters:** `{FILTERS}`
**Script:** `scripts/audit/search_openalex.py` · **Shuffle seed:** {SEED} (numpy default_rng, one generator per pool; pools sorted by OpenAlex ID before shuffling)

| | S1 general | S2 decomposition |
|---|---|---|
| Records reported by API | {counts['S1'][0]} | {counts['S2'][0]} |
| Records downloaded | {counts['S1'][1]} | {counts['S2'][1]} |
| Unique after deduplication (records in both pools assigned to S2) | {counts['S1_unique']} | {counts['S2_unique']} |

- Total downloaded: {n_raw}; unique OpenAlex IDs: {n_after_id}; in both pools: {in_both}.
- Records with an OpenAlex abstract: S1 {int((frames[0].abstract != '').sum())}, S2 {int((frames[1].abstract != '').sum())}.

**S1:** `{STRINGS['S1']}`

**S2:** `{STRINGS['S2']}`
""")
    print(open(os.path.join(OUT, 'search_log.md'), encoding='utf-8').read())


if __name__ == '__main__':
    main()
