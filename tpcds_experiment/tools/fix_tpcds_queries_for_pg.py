#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path


def replace_day_intervals(sql: str) -> str:
    sql = re.sub(r"\+\s*([0-9]+)\s+days\b", r"+ interval '\1 days'", sql, flags=re.IGNORECASE)
    sql = re.sub(r"-\s*([0-9]+)\s+days\b", r"- interval '\1 days'", sql, flags=re.IGNORECASE)
    return sql


def _is_ident_char(ch: str) -> bool:
    return ch.isalnum() or ch == '_'


def _skip_space(sql: str, index: int) -> int:
    length = len(sql)
    while index < length:
        if sql.startswith('--', index):
            newline = sql.find('\n', index)
            if newline == -1:
                return length
            index = newline + 1
            continue
        if sql.startswith('/*', index):
            end = sql.find('*/', index + 2)
            if end == -1:
                return length
            index = end + 2
            continue
        if sql[index].isspace():
            index += 1
            continue
        break
    return index


class _TokenMatch:
    def __init__(self, start: int, end: int):
        self._start = start
        self._end = end

    def start(self) -> int:
        return self._start

    def end(self) -> int:
        return self._end


def add_missing_subquery_aliases(sql: str) -> str:
    tokens = ('from', 'join', ',')
    pieces: list[str] = []
    index = 0
    alias_counter = 1
    length = len(sql)

    while index < length:
        match = None
        for token in tokens:
            if token == ',':
                pos = sql.find(',', index)
                candidate = _TokenMatch(pos, pos + 1) if pos != -1 else None
            else:
                candidate = re.compile(rf'\b{token}\b', re.IGNORECASE).search(sql, index)
            if candidate and (match is None or candidate.start() < match.start()):
                match = candidate
        if match is None:
            pieces.append(sql[index:])
            break

        pieces.append(sql[index:match.end()])
        cursor = _skip_space(sql, match.end())
        pieces.append(sql[match.end():cursor])
        if cursor >= length or sql[cursor] != '(':
            index = cursor
            continue

        inner = _skip_space(sql, cursor + 1)
        inner_end = inner
        while inner_end < length and _is_ident_char(sql[inner_end]):
            inner_end += 1
        first_word = sql[inner:inner_end].lower()
        if first_word not in {'select', 'with'}:
            index = cursor
            continue

        start = cursor
        depth = 0
        i = start
        in_single = False
        while i < length:
            ch = sql[i]
            if in_single:
                if ch == "'":
                    if i + 1 < length and sql[i + 1] == "'":
                        i += 2
                        continue
                    in_single = False
                i += 1
                continue
            if sql.startswith('--', i):
                newline = sql.find('\n', i)
                if newline == -1:
                    i = length
                    break
                i = newline + 1
                continue
            if sql.startswith('/*', i):
                end = sql.find('*/', i + 2)
                if end == -1:
                    i = length
                    break
                i = end + 2
                continue
            if ch == "'":
                in_single = True
                i += 1
                continue
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
                if depth == 0:
                    i += 1
                    break
            i += 1

        pieces.append(sql[start:i])
        lookahead = _skip_space(sql, i)
        token_start = lookahead
        while lookahead < length and _is_ident_char(sql[lookahead]):
            lookahead += 1
        next_word = sql[token_start:lookahead].lower()
        needs_alias = False
        if token_start >= length:
            needs_alias = True
        elif sql[token_start] in ',);':
            needs_alias = True
        elif next_word in {'where', 'group', 'having', 'order', 'limit', 'union', 'intersect', 'except'}:
            needs_alias = True
        elif next_word == '':
            needs_alias = True
        elif next_word == 'as':
            needs_alias = False
        else:
            needs_alias = False

        if needs_alias:
            pieces.append(f' as derived_{alias_counter}')
            alias_counter += 1
        index = i

    return ''.join(pieces)


def replace_lochierarchy_alias(sql: str) -> str:
    match = re.search(r'(?P<expr>grouping\([^\n]+?\)\s*\+\s*grouping\([^\n]+?\))\s+as\s+lochierarchy', sql, flags=re.IGNORECASE)
    if not match:
        return sql
    expr = match.group('expr').strip()
    pattern = re.compile(r'case\s+when\s+lochierarchy\s*=\s*0\s+then', flags=re.IGNORECASE)
    return pattern.sub(f'case when {expr} = 0 then', sql)


def restore_scalar_average_expr(sql: str) -> str:
    pattern = re.compile(r',\s*([A-Za-z_][A-Za-z0-9_+\-*/ ]+)\)/\s*([0-9]+(?:\.[0-9]+)?)\s+average', re.IGNORECASE)
    return pattern.sub(r',(\1)/\2 average', sql)


def strip_template_comments(sql: str) -> str:
    lines = []
    for line in sql.splitlines():
        stripped = line.strip().lower()
        if stripped.startswith('-- start query') or stripped.startswith('-- end query'):
            continue
        lines.append(line)
    return '\n'.join(lines).strip()


def split_statements(sql: str) -> list[str]:
    sql = strip_template_comments(sql)
    statements: list[str] = []
    start = 0
    i = 0
    depth = 0
    in_single = False
    length = len(sql)
    while i < length:
        if in_single:
            if sql[i] == "'":
                if i + 1 < length and sql[i + 1] == "'":
                    i += 2
                    continue
                in_single = False
            i += 1
            continue
        if sql.startswith('--', i):
            newline = sql.find('\n', i)
            if newline == -1:
                break
            i = newline + 1
            continue
        if sql.startswith('/*', i):
            end = sql.find('*/', i + 2)
            if end == -1:
                break
            i = end + 2
            continue
        if sql[i] == "'":
            in_single = True
            i += 1
            continue
        if sql[i] == '(':
            depth += 1
        elif sql[i] == ')':
            if depth > 0:
                depth -= 1
        elif sql[i] == ';' and depth == 0:
            statement = sql[start:i].strip()
            if statement:
                statements.append(statement + ';')
            start = i + 1
        i += 1
    tail = sql[start:].strip()
    if tail:
        statements.append(tail if tail.endswith(';') else tail + ';')
    return statements


def fix_sql(sql: str) -> str:
    fixed = sql
    fixed = replace_day_intervals(fixed)
    fixed = add_missing_subquery_aliases(fixed)
    fixed = replace_lochierarchy_alias(fixed)
    fixed = restore_scalar_average_expr(fixed)
    return fixed


def output_name(base_stem: str, position: int, total: int) -> str:
    if total == 1:
        return f'{base_stem}.sql'
    suffix = chr(ord('a') + position)
    return f'{base_stem}{suffix}.sql'


def main() -> None:
    parser = argparse.ArgumentParser(description='Fix generated TPC-DS SQL for PostgreSQL.')
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for old in output_dir.glob('*.sql'):
        old.unlink()

    changed = 0
    emitted = 0
    source_files = 0
    for path in sorted(input_dir.glob('*.sql')):
        source_files += 1
        original = path.read_text()
        fixed = fix_sql(original)
        if original != fixed:
            changed += 1
        statements = split_statements(fixed)
        for idx, statement in enumerate(statements):
            emitted += 1
            (output_dir / output_name(path.stem, idx, len(statements))).write_text(statement + '\n')

    print(f'Processed {source_files} source SQL files; changed {changed}; emitted {emitted} runnable SQL files.')


if __name__ == '__main__':
    main()
