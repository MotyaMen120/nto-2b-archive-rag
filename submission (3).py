"""Strict CSV transport validation, independent of permissive score.py diagnostics.

validate_rows(predictions, questions, post_ids) accepts iterable mappings (CSV
values must be strings) and iterable integer corpus IDs. Returns error dictionaries
with row (CSV line, or None), qid, code, message; [] means valid.
No correctness, gold-answer, evidence relevance or date-grounding evaluation.
Only the CLI's Parquet reader requires pyarrow; the validation core is stdlib.
"""
import argparse
import csv
from datetime import date
from decimal import Decimal, InvalidOperation
import re
import sys

SPECIAL = {'promised', 'no_info', 'skip'}
ENUMS = {'status': {'done', 'promised', 'postponed', 'cancelled'},
         'yesno': {'yes', 'no'}, 'choice': {'a', 'b', 'c', 'd'}}
NUMBER = re.compile(r'[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?')


def valid_answer(answer, fmt):
    a = answer.strip().lower()
    if a in SPECIAL:
        return True
    if fmt == 'number':
        try:
            return bool(NUMBER.fullmatch(a)) and Decimal(a).is_finite()
        except InvalidOperation:
            return False
    if fmt == 'date':
        try:
            return bool(re.fullmatch(r'[0-9]{4}-[0-9]{2}-[0-9]{2}', a)) and bool(date.fromisoformat(a))
        except ValueError:
            return False
    return a in ENUMS.get(fmt, set())


def validate_rows(predictions, questions, post_ids):
    """Return structured errors; inputs are not mutated or numerically coerced."""
    errors = []
    def error(row, qid, code, message):
        errors.append(dict(row=row, qid=qid, code=code, message=message))
    expected = {}
    for n, question in enumerate(questions, 2):
        qid, fmt = question.get('qid'), question.get('answer_format')
        if not isinstance(qid, str) or not qid.strip() or fmt not in {*ENUMS, 'number', 'date'}:
            error(n, qid, 'questions_schema', 'Questions require nonempty qid and supported answer_format.')
        elif qid in expected:
            error(n, qid, 'questions_duplicate', 'Duplicate question qid.')
        else:
            expected[qid] = fmt
    known = set(post_ids)
    seen = set()
    for n, row in enumerate(predictions, 2):
        qid = row.get('qid')
        if set(row) != {'qid', 'answer', 'evidence'} or any(not isinstance(row.get(k), str) for k in ('qid', 'answer', 'evidence')):
            error(n, qid, 'fields', 'Expected exactly string fields qid, answer, evidence.')
            continue
        if not qid.strip():
            error(n, qid, 'qid_empty', 'qid must not be empty.')
        if qid in seen:
            error(n, qid, 'duplicate', 'Duplicate prediction qid.')
        seen.add(qid)
        if qid not in expected:
            error(n, qid, 'extra', 'qid is not in questions.')
        elif not valid_answer(row['answer'], expected[qid]):
            error(n, qid, 'answer_format', 'Invalid answer for ' + expected[qid] + '.')
        tokens = row['evidence'].split()
        if len(tokens) > 5:
            error(n, qid, 'evidence_count', 'At most five evidence tokens are allowed.')
        for token in tokens:
            if not re.fullmatch(r'[+-]?[0-9]+', token):
                error(n, qid, 'evidence_integer', 'Evidence must contain integer post IDs.')
            else:
                try:
                    present = int(token) in known
                except ValueError:
                    present = False
                if not present:
                    error(n, qid, 'evidence_unknown', 'Evidence post ID is not in the corpus.')
    for qid in sorted(expected.keys() - seen):
        error(None, qid, 'missing', 'Missing prediction.')
    return errors


def read_csv(path, required, exact=False):
    """Read strings only; reject malformed CSV, duplicate headers and ragged rows."""
    with open(path, newline='', encoding='utf-8-sig') as stream:
        reader = csv.reader(stream, strict=True)
        header = next(reader, [])
        if len(set(header)) != len(header) or any(not h.strip() for h in header) or not set(required) <= set(header) or (exact and set(header) != set(required)):
            raise ValueError(f'{path}: invalid headers; required: {", ".join(required)}')
        rows = []
        for row in reader:
            if len(row) != len(header):
                raise ValueError(f'{path}: line {reader.line_num}: wrong field count')
            rows.append(dict(zip(header, row)))
        return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prediction')
    parser.add_argument('--questions', required=True)
    parser.add_argument('--corpus', required=True)
    args = parser.parse_args(argv)
    try:
        predictions = read_csv(args.prediction, ('qid', 'answer', 'evidence'), exact=True)
        questions = read_csv(args.questions, ('qid', 'answer_format'))
        import pyarrow.parquet as pq
        post_ids = pq.read_table(args.corpus, columns=['post_id']).column('post_id').to_pylist()
        errors = validate_rows(predictions, questions, post_ids)
    except (OSError, ValueError, csv.Error, ImportError) as exc:
        print(f'Validation failed: {exc}', file=sys.stderr)
        return 1
    for error in errors:
        print(f"line {error['row'] or '-'}, qid={error['qid']!r}: {error['message']} [{error['code']}]", file=sys.stderr)
    if errors:
        print(f'Invalid submission: {len(errors)} error(s).', file=sys.stderr)
        return 1
    print(f'Valid submission: {len(predictions)} rows (format only; correctness not evaluated).')
    return 0


if __name__ == '__main__':
    sys.exit(main())
