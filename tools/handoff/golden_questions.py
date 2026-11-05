"""Read-only GoldenQuestions.xlsx test data. Never a runtime knowledge/approval source.

Uses only stdlib ZIP/XML; no Excel installation, openpyxl, database or model calls.
The spreadsheet's simulated wording/reference answer and authorized package facts
are separate fields. A differing reference answer is reported, never rewritten.
"""
from collections import Counter
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import json
import re
import zipfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
WORKBOOK = Path('data/knowledge_sets/GOLDEN_QUESTIONS_100/GoldenQuestions.xlsx')
NS = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
REL = '{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id'


def sheets(path):
    """Return literal nonempty cells with row positions; reject formulas in data."""
    with zipfile.ZipFile(path) as archive:
        if any(i.file_size > 20_000_000 for i in archive.infolist()):
            raise ValueError('Workbook XML exceeds read-only size limit')
        strings = []
        if 'xl/sharedStrings.xml' in archive.namelist():
            strings = [''.join(t.text or '' for t in item.findall('.//s:t', NS))
                       for item in ET.fromstring(archive.read('xl/sharedStrings.xml'))]
        rels = {r.attrib['Id']: r.attrib['Target'] for r in
                ET.fromstring(archive.read('xl/_rels/workbook.xml.rels'))}
        result = {}
        for sheet in ET.fromstring(archive.read('xl/workbook.xml')).findall('s:sheets/s:sheet', NS):
            target = rels[sheet.attrib[REL]]
            part = target.lstrip('/') if target.startswith('/') else str(PurePosixPath('xl') / target)
            rows = []
            for row in ET.fromstring(archive.read(part)).findall('s:sheetData/s:row', NS):
                values = {}
                for cell in row:
                    col = re.match(r'[A-Z]+', cell.attrib['r'])[0]
                    typ, value = cell.attrib.get('t'), cell.find('s:v', NS)
                    if int(row.attrib['r']) > 7 and cell.find('s:f', NS) is not None:
                        raise ValueError(f"Formula not permitted in test bank: {sheet.attrib['name']}!{cell.attrib['r']}")
                    if typ == 'e':
                        raise ValueError(f"Excel error: {cell.attrib['r']}")
                    if typ == 's':
                        text = strings[int(value.text)]
                    elif typ == 'inlineStr':
                        text = ''.join(t.text or '' for t in cell.findall('.//s:t', NS))
                    else:
                        text = value.text if value is not None else ''
                    if text:
                        values[col] = text
                if values:
                    rows.append((int(row.attrib['r']), values))
            result[sheet.attrib['name']] = rows
        return result


def table(rows, required):
    headers = next(((n, vals) for n, vals in rows if set(required) <= set(vals.values())), None)
    if headers is None:
        raise ValueError('Missing workbook columns: ' + ', '.join(required))
    number, header = headers
    return [(n, {title: vals.get(col, '') for col, title in header.items()})
            for n, vals in rows if n > number and vals.get('A')]


def read_cases(root=ROOT, workbook=None):
    root = Path(root)
    book = Path(workbook) if workbook else root / WORKBOOK
    tabs = sheets(book)
    main = table(tabs['100组模拟问答'], ['编号', '楼盘', '模拟问题', '模拟回答'])
    evidence = table(tabs['来源与本体关联'], ['编号', '楼盘', '规则 ID', '规则版本', '原文证据'])
    if len(main) != 100 or len(evidence) != 100:
        raise ValueError('Golden Questions must contain exactly 100 paired rows')
    refs = {r['编号']: (n, r) for n, r in evidence}
    if len(refs) != 100 or len({r['编号'] for _, r in main}) != 100:
        raise ValueError('Duplicate case IDs')
    if set(refs) != {r['编号'] for _, r in main}:
        raise ValueError('Question/reference sheets do not have the same case IDs')
    cases = []
    for n, row in main:
        en, ref = refs[row['编号']]
        estate = re.match(r'^E00[12]\b', row['楼盘'])
        if not estate or estate[0] != ref['楼盘'] or not ref['规则 ID'].startswith(estate[0] + ':POL:'):
            raise ValueError('Estate/reference mismatch: ' + row['编号'])
        if not row['模拟问题'].strip() or not row['模拟回答'].strip() or not ref['原文证据'].strip():
            raise ValueError('Missing question/answer/evidence: ' + row['编号'])
        if row['来源文件'] != ref['来源文件'] or row['原文位置'] != ref['原文位置']:
            raise ValueError('Main/reference source mismatch: ' + row['编号'])
        cases.append({'case_id': row['编号'], 'estate_id': estate[0],
                      'simulated_question': row['模拟问题'], 'reference_answer': row['模拟回答'],
                      'expected_policy_id': ref['规则 ID'], 'expected_version': ref['规则版本'],
                      'source_file': row['来源文件'], 'source_locator': row['原文位置'],
                      'policy_statement': ref['业务规则原文'], 'source_excerpt': ref['原文证据'],
                      'evidence_ids': ref['证据 ID'].splitlines(), 'document_ids': ref['文件 ID'].splitlines(),
                      'review_ids': ref['审核记录 ID'].splitlines(), 'usage_note': ref['使用授权'],
                      'source_sheet': '100组模拟问答', 'source_row': n,
                      'evidence_sheet': '来源与本体关联', 'evidence_row': en})
    if Counter(c['estate_id'] for c in cases) != {'E001': 50, 'E002': 50}:
        raise ValueError('Expected 50 cases per estate')
    return cases


def bound_cases(root=ROOT, workbook=None):
    """Join test expectations to separately authorized runtime facts, fail closed."""
    from property_ontology.qa.day1_poc import load
    cases = read_cases(root, workbook)
    packet = load(root)
    if packet is None or packet[2]['active_set'] != 'GOLDEN_QUESTIONS_100':
        raise ValueError('Current authorized Ontology package must be configured')
    rows, receipt, _ = packet
    catalog = {r.get('policy_id', r['qa_id']): r for records in rows.values() for r in records}
    if set(c['expected_policy_id'] for c in cases) != set(catalog):
        raise ValueError('Workbook policy coverage differs from current authorized 100')
    result = []
    for c in cases:
        row = catalog[c['expected_policy_id']]
        proof = receipt['records'][row['qa_id']]
        refs = proof['evidence_refs']
        checks = [c['estate_id'] == row['estate_id'], c['expected_version'] == proof['record_sha256'],
                  c['policy_statement'] == row['answer'].split('\n适用说明：')[0],
                  c['source_excerpt'].strip() == proof['raw_excerpt'].strip(),
                  Path(c['source_file']).name == Path(row['source_file']).name,
                  set(c['evidence_ids']) == {e['chunk_id'] for e in refs},
                  set(c['document_ids']) == {e['document_id'] for e in refs},
                  set(c['review_ids']) <= {e['review_decision_id'] for e in proof['review_events']}]
        if not all(checks):
            raise ValueError(f"Workbook/source mapping invalid for {c['case_id']}: {checks}")
        def display_location(loc):
            if loc['kind'] == 'DOCX_XML':
                return 'DOCX 正文段落 ' + '、'.join(str(x['paragraph_index']) for x in loc['parts'])
            if loc['kind'] == 'PDF_TABLE':
                return f"PDF 第{loc['page']}页；表{loc['table']}；行" + '、'.join(str(x) for x in loc['rows'])
            raise ValueError('Unsupported source location: ' + loc['kind'])
        loc = json.loads(row['source_locator'])
        if set(c['source_locator'].splitlines()) != {display_location(e['locator']) for e in refs}:
            raise ValueError('Workbook location differs from original evidence: ' + c['case_id'])
        result.append({**row, **c, 'authorized_source_path': row['source_file'], 'authorized_source_locator': loc, 'reference_answer_matches_authorized': c['reference_answer'] == row['answer']})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    cases = bound_cases()
    result = {'workbook': str(WORKBOOK), 'sha256': hashlib.sha256((ROOT/WORKBOOK).read_bytes()).hexdigest(),
              'questions': len(cases), 'estates': dict(Counter(c['estate_id'] for c in cases)),
              'valid_policy_mappings': len(cases),
              'reference_answer_exact': sum(c['reference_answer_matches_authorized'] for c in cases),
              'reference_answer_differences': [c for c in cases if not c['reference_answer_matches_authorized']],
              'database_written': False, 'model_requests': 0,
              'scope': 'Spreadsheet/source mapping only; not an Agent accuracy measurement'}
    if args.output:
        if args.output.exists():
            raise SystemExit('Output exists; refuse overwriting evidence')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k != 'reference_answer_differences'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
