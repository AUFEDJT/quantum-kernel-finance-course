"""Verify and stage the authorized 2026-11-19 frozen inputs without network access."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

PANEL_SHA = '3e42c4cedcb4eb43aed05c3cb1f9a74061f8ab339df7c507478dfb094e9b2565'
LGB_SHA = '06d4d9cb2d326195c0c0692b6dce92902037c89cdf0e9081dc8ec3b1edd9782d'
REFERENCE_SHA = 'ac86eb6fe5aa7d5166f2cf906c99c126ae4e7ddde36410af05b6f4546d0b30b5'

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()

def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--repo', type=Path, required=True)
    cli.add_argument('--work', type=Path)
    cli.add_argument('--verify-only', action='store_true')
    args = cli.parse_args()
    repo = args.repo.resolve()
    base = repo / 'private/classic_lab/whale_research'
    panel = repo / 'private/archive/课程可行性验证_20261109/data/daily_panel.csv'
    admission = json.loads((base / 'admission.json').read_text(encoding='utf-8'))
    receipt = json.loads((base / 'lgb_input_receipt.json').read_text(encoding='utf-8'))
    assert admission['asof'] == '2026-11-19', 'This replay profile freezes the original run date.'
    snapshots = []
    for entry in admission['snapshots']:
        name = entry['file']
        assert Path(name).name == name, 'Snapshot must be a simple filename.'
        source = base / name
        assert digest(source) == entry['sha256'], 'Snapshot hash mismatch: ' + name
        snapshots.append(source)
    assert digest(panel) == PANEL_SHA, 'Frozen 20-stock panel differs from the migration manifest.'
    with panel.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    required = {'symbol', 'date', 'close', 'prev_close', 'total_market_cap'}
    assert rows and required.issubset(rows[0]), 'Panel schema incomplete.'
    assert len(rows) == 29086 and len({r['symbol'] for r in rows}) == 20
    assert not receipt['issues'] and receipt['universe'] == receipt['matched'] == 3186
    assert digest(base / 'lgb_inputs.zip') == receipt['sha256'] == LGB_SHA
    member = 'results/wavelet_independent_quotes.csv'
    with tarfile.open(base / 'terminal-lab-evidence.tar.gz', 'r:gz') as archive:
        reference = archive.extractfile(member)
        assert reference is not None, 'Independent reference is absent from the evidence archive.'
        reference_bytes = reference.read()
    assert hashlib.sha256(reference_bytes).hexdigest() == REFERENCE_SHA
    verification = {'profile_date': '2026-11-19', 'network_access': False,
                    'snapshots_verified': len(snapshots), 'panel_rows': len(rows),
                    'panel_symbols': 20, 'panel_sha256': PANEL_SHA,
                    'lgb_input_sha256': LGB_SHA, 'reference_sha256': REFERENCE_SHA}
    if not args.verify_only:
        if args.work is None:
            cli.error('--work is required unless --verify-only is used')
        work = args.work.resolve()
        assert work.is_relative_to((repo / 'private').resolve()), 'Use a work directory under COURSE_ROOT/private.'
        assert not work.exists() or (work.is_dir() and not any(work.iterdir())), 'Work directory must be new or empty.'
        (work / 'inputs').mkdir(parents=True)
        (work / 'results').mkdir()
        for source in [base / 'admission.json', *snapshots]:
            shutil.copyfile(source, work / 'inputs' / source.name)
        for name in ['lgb_inputs.zip', 'lgb_input_receipt.json']:
            shutil.copyfile(base / name, work / name)
        shutil.copyfile(panel, work / 'panel.csv')
        (work / member).write_bytes(reference_bytes)
        (work / 'replay_input_receipt.json').write_text(json.dumps(verification, indent=2), encoding='utf-8')
    print(json.dumps(verification, indent=2))

if __name__ == '__main__':
    main()