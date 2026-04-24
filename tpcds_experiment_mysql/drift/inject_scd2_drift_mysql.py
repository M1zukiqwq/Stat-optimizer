#!/usr/bin/env python3
"""Inject SCD2-style drift into TPC-DS dimensions on MySQL 8."""

from __future__ import annotations

import argparse
from decimal import Decimal
from datetime import timedelta
from typing import Any, Dict, Iterable, Sequence

import pymysql
from pymysql.cursors import DictCursor


DRIFT_ROUNDS = 10
DRIFT_RATIO = 0.05
ITEM_FACTS_PER_KEY = 10
CUSTOMER_FACTS_PER_KEY = 5
DATE_SHIFT_DAYS = 30
PRICE_DRIFT_FACTOR = Decimal('1.1')

ITEM_INSERT_SQL = """
INSERT INTO item (
    i_item_sk, i_item_id, i_rec_start_date, i_rec_end_date,
    i_item_desc, i_current_price, i_wholesale_cost, i_brand_id,
    i_brand, i_class_id, i_class, i_category_id, i_category,
    i_manufact_id, i_manufact, i_size, i_formulation, i_color,
    i_units, i_container, i_manager_id, i_product_name
) VALUES (
    %s, %s, %s, %s,
    %s, %s, %s, %s,
    %s, %s, %s, %s, %s,
    %s, %s, %s, %s, %s,
    %s, %s, %s, %s
)
"""

STORE_SALES_INSERT_SQL = """
INSERT INTO store_sales (
    ss_sold_date_sk, ss_sold_time_sk, ss_item_sk, ss_customer_sk,
    ss_cdemo_sk, ss_hdemo_sk, ss_addr_sk, ss_store_sk, ss_promo_sk,
    ss_ticket_number, ss_quantity, ss_wholesale_cost, ss_list_price,
    ss_sales_price, ss_ext_discount_amt, ss_ext_sales_price,
    ss_ext_wholesale_cost, ss_ext_list_price, ss_ext_tax,
    ss_coupon_amt, ss_net_paid, ss_net_paid_inc_tax, ss_net_profit
) VALUES (
    %s, %s, %s, %s,
    %s, %s, %s, %s, %s,
    %s, %s, %s, %s,
    %s, %s, %s,
    %s, %s, %s,
    %s, %s, %s, %s
)
"""


def batch_size(total_rows: int, drift_ratio: float) -> int:
    if total_rows <= 0:
        return 0
    return max(1, int(total_rows * drift_ratio))


def column_exists(cur, table_name: str, column_name: str) -> bool:
    cur.execute(
        """
        SELECT COUNT(*) AS cnt
        FROM information_schema.columns
        WHERE table_schema = DATABASE()
          AND table_name = %s
          AND column_name = %s
        """,
        (table_name, column_name),
    )
    return cur.fetchone()['cnt'] > 0


def index_exists(cur, table_name: str, index_name: str) -> bool:
    cur.execute(
        """
        SELECT COUNT(*) AS cnt
        FROM information_schema.statistics
        WHERE table_schema = DATABASE()
          AND table_name = %s
          AND index_name = %s
        """,
        (table_name, index_name),
    )
    return cur.fetchone()['cnt'] > 0


def customer_review_date_column(cur) -> str:
    if column_exists(cur, 'customer', 'c_last_review_date_sk'):
        return 'c_last_review_date_sk'
    if column_exists(cur, 'customer', 'c_last_review_date'):
        return 'c_last_review_date'
    raise RuntimeError(
        'customer table is missing both c_last_review_date_sk and c_last_review_date'
    )


def ensure_customer_scd2_columns(cur) -> None:
    if (
        not column_exists(cur, 'customer', 'c_last_review_date_sk')
        and column_exists(cur, 'customer', 'c_last_review_date')
    ):
        cur.execute("ALTER TABLE customer ADD COLUMN c_last_review_date_sk INTEGER NULL")
        cur.execute(
            """
            UPDATE customer c
            JOIN date_dim dd ON dd.d_date = DATE(c.c_last_review_date)
            SET c.c_last_review_date_sk = dd.d_date_sk
            WHERE c.c_last_review_date IS NOT NULL
              AND c.c_last_review_date_sk IS NULL
            """
        )
        if not index_exists(cur, 'customer', 'customer_last_review_date_sk_idx'):
            cur.execute(
                "CREATE INDEX customer_last_review_date_sk_idx ON customer (c_last_review_date_sk)"
            )

    if not column_exists(cur, 'customer', 'c_rec_start_date'):
        cur.execute("ALTER TABLE customer ADD COLUMN c_rec_start_date DATE NULL")
    if not column_exists(cur, 'customer', 'c_rec_end_date'):
        cur.execute("ALTER TABLE customer ADD COLUMN c_rec_end_date DATE NULL")

    cur.execute(
        """
        UPDATE customer
        SET c_rec_start_date = COALESCE(c_rec_start_date, CURDATE())
        WHERE c_rec_start_date IS NULL
        """
    )

    cur.execute("DROP TEMPORARY TABLE IF EXISTS customer_latest")
    cur.execute(
        """
        CREATE TEMPORARY TABLE customer_latest AS
        SELECT c_customer_id, MAX(c_customer_sk) AS c_customer_sk
        FROM customer
        GROUP BY c_customer_id
        """
    )
    cur.execute(
        """
        UPDATE customer c
        JOIN customer_latest latest ON latest.c_customer_id = c.c_customer_id
        SET c.c_rec_end_date = CURDATE() - INTERVAL 1 DAY
        WHERE c.c_customer_sk <> latest.c_customer_sk
          AND c.c_rec_end_date IS NULL
        """
    )

    if not index_exists(cur, 'customer', 'customer_rec_end_date_idx'):
        cur.execute("CREATE INDEX customer_rec_end_date_idx ON customer (c_rec_end_date)")
    if not index_exists(cur, 'customer', 'customer_id_rec_end_idx'):
        cur.execute("CREATE INDEX customer_id_rec_end_idx ON customer (c_customer_id, c_rec_end_date)")


def ensure_item_current_rows(cur) -> None:
    cur.execute("DROP TEMPORARY TABLE IF EXISTS item_latest")
    cur.execute(
        """
        CREATE TEMPORARY TABLE item_latest AS
        SELECT i_item_id, MAX(i_item_sk) AS i_item_sk
        FROM item
        GROUP BY i_item_id
        """
    )
    cur.execute(
        """
        UPDATE item i
        JOIN item_latest latest
          ON latest.i_item_id = i.i_item_id
         AND latest.i_item_sk = i.i_item_sk
        SET i.i_rec_end_date = NULL
        WHERE i.i_rec_end_date IS NOT NULL
        """
    )


def prepare_session_state(cur) -> None:
    ensure_item_current_rows(cur)
    ensure_customer_scd2_columns(cur)
    cur.execute("DROP TEMPORARY TABLE IF EXISTS donor_store_sales")
    cur.execute("DROP TEMPORARY TABLE IF EXISTS donor_item_map")
    cur.execute("DROP TEMPORARY TABLE IF EXISTS donor_customer_map")
    cur.execute("DROP TEMPORARY TABLE IF EXISTS customer_current_map")
    cur.execute(
        """
        CREATE TEMPORARY TABLE donor_store_sales AS
        SELECT * FROM store_sales
        """
    )
    cur.execute(
        """
        CREATE TEMPORARY TABLE donor_item_map AS
        SELECT i.i_item_id, MIN(ss.ss_item_sk) AS donor_item_sk
        FROM donor_store_sales ss
        JOIN item i ON i.i_item_sk = ss.ss_item_sk
        WHERE i.i_rec_end_date IS NULL
        GROUP BY i.i_item_id
        """
    )
    cur.execute(
        """
        CREATE TEMPORARY TABLE customer_current_map AS
        SELECT c_customer_id, c_customer_sk
        FROM customer
        WHERE c_rec_end_date IS NULL
        """
    )
    cur.execute(
        """
        CREATE TEMPORARY TABLE donor_customer_map AS
        SELECT c.c_customer_id, MIN(ss.ss_customer_sk) AS donor_customer_sk
        FROM donor_store_sales ss
        JOIN customer c ON c.c_customer_sk = ss.ss_customer_sk
        JOIN customer_current_map cur ON cur.c_customer_id = c.c_customer_id
                                     AND cur.c_customer_sk = c.c_customer_sk
        GROUP BY c.c_customer_id
        """
    )


def load_date_maps(cur) -> tuple[dict[int, Any], dict[Any, int]]:
    cur.execute("SELECT d_date_sk, d_date FROM date_dim")
    rows = cur.fetchall()
    by_sk = {row['d_date_sk']: row['d_date'] for row in rows}
    by_date = {row['d_date']: row['d_date_sk'] for row in rows}
    return by_sk, by_date


def shift_date_sk(date_by_sk: dict[int, Any], sk_by_date: dict[Any, int], sold_date_sk: int, date_shift_days: int) -> int:
    old_date = date_by_sk.get(sold_date_sk)
    if old_date is None:
        return sold_date_sk
    new_sk = sk_by_date.get(old_date + timedelta(days=date_shift_days))
    return new_sk or sold_date_sk


def fetch_rows_by_keys(cur, table: str, column: str, keys: Sequence[int]) -> list[Dict[str, Any]]:
    if not keys:
        return []
    placeholders = ','.join(['%s'] * len(keys))
    cur.execute(f"SELECT * FROM {table} WHERE {column} IN ({placeholders})", tuple(keys))
    return list(cur.fetchall())


def update_rows_end_date(cur, table: str, column: str, keys: Sequence[int], end_expr: str) -> None:
    if not keys:
        return
    placeholders = ','.join(['%s'] * len(keys))
    cur.execute(f"UPDATE {table} SET {end_expr} WHERE {column} IN ({placeholders})", tuple(keys))


def fetch_donor_sales(cur, key_column: str, key_value: int, limit_rows: int) -> list[Dict[str, Any]]:
    cur.execute(
        f"SELECT * FROM donor_store_sales WHERE {key_column} = %s ORDER BY ss_ticket_number LIMIT %s",
        (key_value, limit_rows),
    )
    return list(cur.fetchall())


def fetch_donor_sales_batch(cur, key_column: str, key_values: Sequence[int], limit_rows: int) -> dict[int, list[Dict[str, Any]]]:
    if not key_values:
        return {}
    placeholders = ','.join(['%s'] * len(key_values))
    cur.execute(
        f"""
        SELECT *
        FROM (
            SELECT donor_store_sales.*, ROW_NUMBER() OVER (PARTITION BY {key_column} ORDER BY ss_ticket_number) AS rn
            FROM donor_store_sales
            WHERE {key_column} IN ({placeholders})
        ) ranked_sales
        WHERE rn <= %s
        ORDER BY {key_column}, ss_ticket_number
        """,
        tuple(key_values) + (limit_rows,),
    )
    grouped: dict[int, list[Dict[str, Any]]] = {}
    for row in cur.fetchall():
        row.pop('rn', None)
        grouped.setdefault(row[key_column], []).append(row)
    return grouped


def inject_item_round(cur, drift_ratio: float, facts_per_key: int, date_shift_days: int, date_by_sk, sk_by_date) -> tuple[int, int]:
    cur.execute(
        """
        SELECT COUNT(*) AS cnt
        FROM item i
        JOIN donor_item_map d ON d.i_item_id = i.i_item_id
        WHERE i.i_rec_end_date IS NULL
        """
    )
    total = cur.fetchone()['cnt']
    current_batch = batch_size(total, drift_ratio)
    if current_batch == 0:
        return 0, 0

    cur.execute("SELECT COALESCE(MAX(i_item_sk), 0) AS max_sk FROM item")
    max_item_sk = cur.fetchone()['max_sk']
    cur.execute(
        """
        SELECT i.i_item_sk AS old_item_sk, i.i_item_id, d.donor_item_sk
        FROM item i
        JOIN donor_item_map d ON d.i_item_id = i.i_item_id
        WHERE i.i_rec_end_date IS NULL
        ORDER BY RAND()
        LIMIT %s
        """,
        (current_batch,),
    )
    selected = list(cur.fetchall())
    if not selected:
        return 0, 0

    mappings = []
    for offset, row in enumerate(selected, start=1):
        mappings.append({
            'old_item_sk': row['old_item_sk'],
            'i_item_id': row['i_item_id'],
            'donor_item_sk': row['donor_item_sk'],
            'new_item_sk': max_item_sk + offset,
        })

    old_item_sks = [row['old_item_sk'] for row in mappings]
    update_rows_end_date(cur, 'item', 'i_item_sk', old_item_sks, 'i_rec_end_date = CURDATE() - INTERVAL 1 DAY')

    source_rows = {row['i_item_sk']: row for row in fetch_rows_by_keys(cur, 'item', 'i_item_sk', old_item_sks)}
    item_rows = []
    for mapping in mappings:
        src = source_rows[mapping['old_item_sk']]
        item_rows.append((
            mapping['new_item_sk'],
            src['i_item_id'],
            None,
            src['i_item_desc'],
            src['i_current_price'] * PRICE_DRIFT_FACTOR,
            src['i_wholesale_cost'] * PRICE_DRIFT_FACTOR,
            src['i_brand_id'],
            src['i_brand'],
            src['i_class_id'],
            src['i_class'],
            src['i_category_id'],
            src['i_category'],
            src['i_manufact_id'],
            src['i_manufact'],
            src['i_size'],
            src['i_formulation'],
            src['i_color'],
            src['i_units'],
            src['i_container'],
            src['i_manager_id'],
            src['i_product_name'],
        ))
    cur.executemany(
        """
        INSERT INTO item (
            i_item_sk, i_item_id, i_rec_start_date, i_rec_end_date,
            i_item_desc, i_current_price, i_wholesale_cost, i_brand_id,
            i_brand, i_class_id, i_class, i_category_id, i_category,
            i_manufact_id, i_manufact, i_size, i_formulation, i_color,
            i_units, i_container, i_manager_id, i_product_name
        ) VALUES (
            %s, %s, CURDATE(), %s,
            %s, %s, %s, %s,
            %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s,
            %s, %s, %s, %s
        )
        """,
        item_rows,
    )
    inserted_items = cur.rowcount

    cur.execute("SELECT COALESCE(MAX(ss_ticket_number), 0) AS max_ticket FROM store_sales")
    max_ticket = cur.fetchone()['max_ticket']
    next_ticket = max_ticket
    sales_rows = []
    donor_sales_by_item = fetch_donor_sales_batch(
        cur,
        'ss_item_sk',
        [mapping['donor_item_sk'] for mapping in mappings],
        facts_per_key,
    )
    for mapping in mappings:
        for donor in donor_sales_by_item.get(mapping['donor_item_sk'], []):
            next_ticket += 1
            sales_rows.append((
                shift_date_sk(date_by_sk, sk_by_date, donor['ss_sold_date_sk'], date_shift_days),
                donor['ss_sold_time_sk'],
                mapping['new_item_sk'],
                donor['ss_customer_sk'],
                donor['ss_cdemo_sk'],
                donor['ss_hdemo_sk'],
                donor['ss_addr_sk'],
                donor['ss_store_sk'],
                donor['ss_promo_sk'],
                next_ticket,
                donor['ss_quantity'],
                donor['ss_wholesale_cost'],
                donor['ss_list_price'],
                donor['ss_sales_price'],
                donor['ss_ext_discount_amt'],
                donor['ss_ext_sales_price'],
                donor['ss_ext_wholesale_cost'],
                donor['ss_ext_list_price'],
                donor['ss_ext_tax'],
                donor['ss_coupon_amt'],
                donor['ss_net_paid'],
                donor['ss_net_paid_inc_tax'],
                donor['ss_net_profit'],
            ))
    if sales_rows:
        cur.executemany(STORE_SALES_INSERT_SQL, sales_rows)
    return inserted_items, len(sales_rows)


def inject_customer_round(cur, drift_ratio: float, facts_per_key: int, date_shift_days: int, date_by_sk, sk_by_date) -> tuple[int, int]:
    review_date_column = customer_review_date_column(cur)
    cur.execute(
        """
        SELECT COUNT(*) AS cnt
        FROM customer_current_map c
        JOIN donor_customer_map d ON d.c_customer_id = c.c_customer_id
        """
    )
    total = cur.fetchone()['cnt']
    current_batch = batch_size(total, drift_ratio)
    if current_batch == 0:
        return 0, 0

    cur.execute("SELECT COALESCE(MAX(c_customer_sk), 0) AS max_sk FROM customer")
    max_customer_sk = cur.fetchone()['max_sk']
    cur.execute(
        """
        SELECT c.c_customer_id, c.c_customer_sk AS old_customer_sk, d.donor_customer_sk
        FROM customer_current_map c
        JOIN donor_customer_map d ON d.c_customer_id = c.c_customer_id
        ORDER BY RAND()
        LIMIT %s
        """,
        (current_batch,),
    )
    selected = list(cur.fetchall())
    if not selected:
        return 0, 0

    mappings = []
    for offset, row in enumerate(selected, start=1):
        mappings.append({
            'c_customer_id': row['c_customer_id'],
            'old_customer_sk': row['old_customer_sk'],
            'donor_customer_sk': row['donor_customer_sk'],
            'new_customer_sk': max_customer_sk + offset,
        })

    old_customer_sks = [row['old_customer_sk'] for row in mappings]
    update_rows_end_date(cur, 'customer', 'c_customer_sk', old_customer_sks, 'c_rec_end_date = CURDATE() - INTERVAL 1 DAY')

    source_rows = {row['c_customer_sk']: row for row in fetch_rows_by_keys(cur, 'customer', 'c_customer_sk', old_customer_sks)}
    customer_rows = []
    for mapping in mappings:
        src = source_rows[mapping['old_customer_sk']]
        customer_rows.append((
            mapping['new_customer_sk'],
            src['c_customer_id'],
            src['c_current_cdemo_sk'],
            src['c_current_hdemo_sk'],
            src['c_current_addr_sk'],
            src['c_first_shipto_date_sk'],
            src['c_first_sales_date_sk'],
            src['c_salutation'],
            src['c_first_name'],
            src['c_last_name'],
            src['c_preferred_cust_flag'],
            src['c_birth_day'],
            src['c_birth_month'],
            src['c_birth_year'],
            src['c_birth_country'],
            src['c_login'],
            src['c_email_address'],
            src[review_date_column],
            None,
        ))
    cur.executemany(
        f"""
        INSERT INTO customer (
            c_customer_sk, c_customer_id, c_current_cdemo_sk, c_current_hdemo_sk,
            c_current_addr_sk, c_first_shipto_date_sk, c_first_sales_date_sk,
            c_salutation, c_first_name, c_last_name, c_preferred_cust_flag,
            c_birth_day, c_birth_month, c_birth_year, c_birth_country, c_login,
            c_email_address, {review_date_column}, c_rec_start_date, c_rec_end_date
        ) VALUES (
            %s, %s, %s, %s,
            %s, %s, %s,
            %s, %s, %s, %s,
            %s, %s, %s, %s, %s,
            %s, %s, CURDATE(), %s
        )
        """,
        customer_rows,
    )
    inserted_customers = cur.rowcount

    cur.executemany(
        "UPDATE customer_current_map SET c_customer_sk = %s WHERE c_customer_id = %s",
        [(row['new_customer_sk'], row['c_customer_id']) for row in mappings],
    )

    cur.execute("SELECT COALESCE(MAX(ss_ticket_number), 0) AS max_ticket FROM store_sales")
    max_ticket = cur.fetchone()['max_ticket']
    next_ticket = max_ticket
    sales_rows = []
    donor_sales_by_customer = fetch_donor_sales_batch(
        cur,
        'ss_customer_sk',
        [mapping['donor_customer_sk'] for mapping in mappings],
        facts_per_key,
    )
    for mapping in mappings:
        for donor in donor_sales_by_customer.get(mapping['donor_customer_sk'], []):
            next_ticket += 1
            sales_rows.append((
                shift_date_sk(date_by_sk, sk_by_date, donor['ss_sold_date_sk'], date_shift_days),
                donor['ss_sold_time_sk'],
                donor['ss_item_sk'],
                mapping['new_customer_sk'],
                donor['ss_cdemo_sk'],
                donor['ss_hdemo_sk'],
                donor['ss_addr_sk'],
                donor['ss_store_sk'],
                donor['ss_promo_sk'],
                next_ticket,
                donor['ss_quantity'],
                donor['ss_wholesale_cost'],
                donor['ss_list_price'],
                donor['ss_sales_price'],
                donor['ss_ext_discount_amt'],
                donor['ss_ext_sales_price'],
                donor['ss_ext_wholesale_cost'],
                donor['ss_ext_list_price'],
                donor['ss_ext_tax'],
                donor['ss_coupon_amt'],
                donor['ss_net_paid'],
                donor['ss_net_paid_inc_tax'],
                donor['ss_net_profit'],
            ))
    if sales_rows:
        cur.executemany(STORE_SALES_INSERT_SQL, sales_rows)
    return inserted_customers, len(sales_rows)


def inject_scd2_drift(conn, rounds: int, drift_ratio: float, item_facts_per_key: int, customer_facts_per_key: int, date_shift_days: int) -> None:
    with conn.cursor() as cur:
        prepare_session_state(cur)
        date_by_sk, sk_by_date = load_date_maps(cur)
    conn.commit()

    try:
        for round_no in range(1, rounds + 1):
            with conn.cursor() as cur:
                print(f"\n--- Round {round_no}/{rounds} ---")
                inserted_items, item_sales = inject_item_round(cur, drift_ratio, item_facts_per_key, date_shift_days, date_by_sk, sk_by_date)
                print(f"  item: created {inserted_items} new keys, inserted {item_sales} fact rows.")
                inserted_customers, customer_sales = inject_customer_round(cur, drift_ratio, customer_facts_per_key, date_shift_days, date_by_sk, sk_by_date)
                print(f"  customer: created {inserted_customers} new keys, inserted {customer_sales} fact rows.")
            conn.commit()
    except Exception:
        conn.rollback()
        raise

    print("\nSCD2 drift injection complete.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dbname', default='tpcds')
    parser.add_argument('--user', default='root')
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--port', type=int, default=3306)
    parser.add_argument('--rounds', type=int, default=DRIFT_ROUNDS)
    parser.add_argument('--drift-ratio', type=float, default=DRIFT_RATIO)
    parser.add_argument('--item-facts-per-key', type=int, default=ITEM_FACTS_PER_KEY)
    parser.add_argument('--customer-facts-per-key', type=int, default=CUSTOMER_FACTS_PER_KEY)
    parser.add_argument('--date-shift-days', type=int, default=DATE_SHIFT_DAYS)
    parser.add_argument('--password', default='tianqichu123')
    args = parser.parse_args()

    conn = pymysql.connect(
        host=args.host,
        port=args.port,
        user=args.user,
        password=args.password,
        database=args.dbname,
        autocommit=False,
        cursorclass=DictCursor,
        local_infile=True,
    )
    try:
        inject_scd2_drift(
            conn,
            rounds=args.rounds,
            drift_ratio=args.drift_ratio,
            item_facts_per_key=args.item_facts_per_key,
            customer_facts_per_key=args.customer_facts_per_key,
            date_shift_days=args.date_shift_days,
        )
    finally:
        conn.close()


if __name__ == '__main__':
    main()
