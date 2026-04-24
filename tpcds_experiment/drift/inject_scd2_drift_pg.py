#!/usr/bin/env python3
"""
Inject SCD2-style drift into TPC-DS dimensions and append matching facts.
"""

import argparse

import psycopg2


DRIFT_ROUNDS = 10
DRIFT_RATIO = 0.05
ITEM_FACTS_PER_KEY = 10
CUSTOMER_FACTS_PER_KEY = 5
DATE_SHIFT_DAYS = 30


def batch_size(total_rows: int, drift_ratio: float) -> int:
    if total_rows <= 0:
        return 0
    return max(1, int(total_rows * drift_ratio))


def column_exists(cur, table_name: str, column_name: str) -> bool:
    cur.execute(
        """
        SELECT EXISTS (
            SELECT 1
            FROM information_schema.columns
            WHERE table_schema = current_schema()
              AND table_name = %s
              AND column_name = %s
        )
        """,
        (table_name, column_name),
    )
    return cur.fetchone()[0]


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
        cur.execute(
            "ALTER TABLE customer ADD COLUMN IF NOT EXISTS c_last_review_date_sk integer"
        )
        cur.execute(
            """
            UPDATE customer c
            SET c_last_review_date_sk = dd.d_date_sk
            FROM date_dim dd
            WHERE c.c_last_review_date IS NOT NULL
              AND c.c_last_review_date_sk IS NULL
              AND dd.d_date::text = c.c_last_review_date::text
            """
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS customer_last_review_date_sk_idx ON customer (c_last_review_date_sk)"
        )

    cur.execute(
        """
        ALTER TABLE customer
        ADD COLUMN IF NOT EXISTS c_rec_start_date date,
        ADD COLUMN IF NOT EXISTS c_rec_end_date date
        """
    )
    cur.execute(
        """
        UPDATE customer
        SET c_rec_start_date = COALESCE(c_rec_start_date, CURRENT_DATE),
            c_rec_end_date = CASE WHEN c_rec_end_date IS NOT NULL THEN c_rec_end_date ELSE NULL END
        WHERE c_rec_start_date IS NULL
        """
    )
    cur.execute(
        """
        WITH ranked AS (
            SELECT
                c_customer_sk,
                ROW_NUMBER() OVER (
                    PARTITION BY c_customer_id
                    ORDER BY c_rec_start_date DESC NULLS LAST, c_customer_sk DESC
                ) AS recency_rank
            FROM customer
        )
        UPDATE customer c
        SET c_rec_end_date = CURRENT_DATE - 1
        FROM ranked r
        WHERE c.c_customer_sk = r.c_customer_sk
          AND r.recency_rank > 1
          AND c.c_rec_end_date IS NULL
        """
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS customer_rec_end_date_idx ON customer (c_rec_end_date)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS customer_id_rec_end_idx ON customer (c_customer_id, c_rec_end_date)"
    )


def prepare_session_state(cur) -> None:
    cur.execute("SET max_parallel_workers_per_gather = 0")
    ensure_customer_scd2_columns(cur)
    cur.execute("DROP TABLE IF EXISTS donor_store_sales")
    cur.execute("DROP TABLE IF EXISTS donor_item_map")
    cur.execute("DROP TABLE IF EXISTS donor_customer_map")
    cur.execute("DROP TABLE IF EXISTS customer_current_map")
    cur.execute("""
        CREATE TEMP TABLE donor_store_sales AS
        SELECT * FROM store_sales
    """)
    cur.execute("CREATE INDEX donor_store_sales_item_idx ON donor_store_sales (ss_item_sk)")
    cur.execute("CREATE INDEX donor_store_sales_customer_idx ON donor_store_sales (ss_customer_sk)")
    cur.execute("""
        CREATE TEMP TABLE donor_item_map AS
        SELECT DISTINCT ON (i.i_item_id)
            i.i_item_id,
            ss.ss_item_sk AS donor_item_sk
        FROM donor_store_sales ss
        JOIN item i ON i.i_item_sk = ss.ss_item_sk
        WHERE i.i_rec_end_date IS NULL
        ORDER BY i.i_item_id, ss.ss_ticket_number
    """)
    cur.execute("CREATE UNIQUE INDEX donor_item_map_id_idx ON donor_item_map (i_item_id)")
    cur.execute("CREATE INDEX donor_item_map_sk_idx ON donor_item_map (donor_item_sk)")
    cur.execute("""
        CREATE TEMP TABLE customer_current_map AS
        SELECT c_customer_id, c_customer_sk
        FROM customer
        WHERE c_rec_end_date IS NULL
    """)
    cur.execute("CREATE UNIQUE INDEX customer_current_map_id_idx ON customer_current_map (c_customer_id)")
    cur.execute("CREATE INDEX customer_current_map_sk_idx ON customer_current_map (c_customer_sk)")
    cur.execute("""
        CREATE TEMP TABLE donor_customer_map AS
        SELECT DISTINCT ON (c.c_customer_id)
            c.c_customer_id,
            ss.ss_customer_sk AS donor_customer_sk
        FROM donor_store_sales ss
        JOIN customer c ON c.c_customer_sk = ss.ss_customer_sk
        JOIN customer_current_map cur ON cur.c_customer_id = c.c_customer_id
        WHERE cur.c_customer_sk = c.c_customer_sk
        ORDER BY c.c_customer_id, ss.ss_ticket_number
    """)
    cur.execute("CREATE UNIQUE INDEX donor_customer_map_id_idx ON donor_customer_map (c_customer_id)")
    cur.execute("CREATE INDEX donor_customer_map_sk_idx ON donor_customer_map (donor_customer_sk)")


def inject_item_round(cur, drift_ratio: float, facts_per_key: int, date_shift_days: int) -> tuple[int, int]:
    cur.execute("""
        SELECT COUNT(*)
        FROM item i
        JOIN donor_item_map d ON d.i_item_id = i.i_item_id
        WHERE i.i_rec_end_date IS NULL
    """)
    total = cur.fetchone()[0]
    current_batch = batch_size(total, drift_ratio)
    if current_batch == 0:
        return 0, 0

    cur.execute("SELECT COALESCE(MAX(i_item_sk), 0) FROM item")
    max_item_sk = cur.fetchone()[0]

    cur.execute("DROP TABLE IF EXISTS item_round_map")
    cur.execute(
        """
        CREATE TEMP TABLE item_round_map AS
        WITH selected AS (
            SELECT i.i_item_sk, i.i_item_id, d.donor_item_sk
            FROM item i
            JOIN donor_item_map d ON d.i_item_id = i.i_item_id
            WHERE i.i_rec_end_date IS NULL
            ORDER BY RANDOM()
            LIMIT %s
        )
        SELECT
            i_item_sk AS old_item_sk,
            i_item_id,
            donor_item_sk,
            %s + ROW_NUMBER() OVER (ORDER BY i_item_sk) AS new_item_sk
        FROM selected
        """,
        (current_batch, max_item_sk),
    )

    cur.execute("SELECT COUNT(*) FROM item_round_map")
    selected_count = cur.fetchone()[0]
    if selected_count == 0:
        return 0, 0

    cur.execute(
        """
        UPDATE item
        SET i_rec_end_date = CURRENT_DATE - 1
        WHERE i_item_sk IN (SELECT old_item_sk FROM item_round_map)
        """
    )

    cur.execute(
        """
        INSERT INTO item (
            i_item_sk, i_item_id, i_rec_start_date, i_rec_end_date,
            i_item_desc, i_current_price, i_wholesale_cost, i_brand_id,
            i_brand, i_class_id, i_class, i_category_id, i_category,
            i_manufact_id, i_manufact, i_size, i_formulation, i_color,
            i_units, i_container, i_manager_id, i_product_name
        )
        SELECT
            map.new_item_sk,
            src.i_item_id,
            CURRENT_DATE,
            NULL,
            src.i_item_desc,
            src.i_current_price * 1.1,
            src.i_wholesale_cost * 1.1,
            src.i_brand_id,
            src.i_brand,
            src.i_class_id,
            src.i_class,
            src.i_category_id,
            src.i_category,
            src.i_manufact_id,
            src.i_manufact,
            src.i_size,
            src.i_formulation,
            src.i_color,
            src.i_units,
            src.i_container,
            src.i_manager_id,
            src.i_product_name
        FROM item_round_map map
        JOIN item src ON src.i_item_sk = map.old_item_sk
        ORDER BY map.new_item_sk
        """
    )
    inserted_items = cur.rowcount

    cur.execute("SELECT COALESCE(MAX(ss_ticket_number), 0) FROM store_sales")
    max_ticket = cur.fetchone()[0]
    cur.execute(
        """
        WITH donor_rows AS (
            SELECT
                COALESCE(new_dd.d_date_sk, ss.ss_sold_date_sk) AS sold_date_sk,
                ss.ss_sold_time_sk,
                map.new_item_sk AS item_sk,
                ss.ss_customer_sk,
                ss.ss_cdemo_sk,
                ss.ss_hdemo_sk,
                ss.ss_addr_sk,
                ss.ss_store_sk,
                ss.ss_promo_sk,
                ss.ss_quantity,
                ss.ss_wholesale_cost,
                ss.ss_list_price,
                ss.ss_sales_price,
                ss.ss_ext_discount_amt,
                ss.ss_ext_sales_price,
                ss.ss_ext_wholesale_cost,
                ss.ss_ext_list_price,
                ss.ss_ext_tax,
                ss.ss_coupon_amt,
                ss.ss_net_paid,
                ss.ss_net_paid_inc_tax,
                ss.ss_net_profit,
                ROW_NUMBER() OVER (ORDER BY map.new_item_sk, ss.ss_ticket_number) AS rn
            FROM item_round_map map
            JOIN LATERAL (
                SELECT *
                FROM donor_store_sales ss
                WHERE ss.ss_item_sk = map.donor_item_sk
                ORDER BY ss.ss_ticket_number
                LIMIT %s
            ) ss ON TRUE
            LEFT JOIN date_dim old_dd ON old_dd.d_date_sk = ss.ss_sold_date_sk
            LEFT JOIN date_dim new_dd ON new_dd.d_date = old_dd.d_date + %s
        )
        INSERT INTO store_sales (
            ss_sold_date_sk, ss_sold_time_sk, ss_item_sk, ss_customer_sk,
            ss_cdemo_sk, ss_hdemo_sk, ss_addr_sk, ss_store_sk, ss_promo_sk,
            ss_ticket_number, ss_quantity, ss_wholesale_cost, ss_list_price,
            ss_sales_price, ss_ext_discount_amt, ss_ext_sales_price,
            ss_ext_wholesale_cost, ss_ext_list_price, ss_ext_tax,
            ss_coupon_amt, ss_net_paid, ss_net_paid_inc_tax, ss_net_profit
        )
        SELECT
            sold_date_sk,
            ss_sold_time_sk,
            item_sk,
            ss_customer_sk,
            ss_cdemo_sk,
            ss_hdemo_sk,
            ss_addr_sk,
            ss_store_sk,
            ss_promo_sk,
            %s + rn,
            ss_quantity,
            ss_wholesale_cost,
            ss_list_price,
            ss_sales_price,
            ss_ext_discount_amt,
            ss_ext_sales_price,
            ss_ext_wholesale_cost,
            ss_ext_list_price,
            ss_ext_tax,
            ss_coupon_amt,
            ss_net_paid,
            ss_net_paid_inc_tax,
            ss_net_profit
        FROM donor_rows
        ORDER BY rn
        """,
        (facts_per_key, date_shift_days, max_ticket),
    )
    inserted_sales = cur.rowcount
    return inserted_items, inserted_sales


def inject_customer_round(cur, drift_ratio: float, facts_per_key: int, date_shift_days: int) -> tuple[int, int]:
    review_date_column = customer_review_date_column(cur)
    cur.execute(
        """
        SELECT COUNT(*)
        FROM customer_current_map c
        JOIN donor_customer_map d ON d.c_customer_id = c.c_customer_id
        """
    )
    total = cur.fetchone()[0]
    current_batch = batch_size(total, drift_ratio)
    if current_batch == 0:
        return 0, 0

    cur.execute("SELECT COALESCE(MAX(c_customer_sk), 0) FROM customer")
    max_customer_sk = cur.fetchone()[0]

    cur.execute("DROP TABLE IF EXISTS customer_round_map")
    cur.execute(
        """
        CREATE TEMP TABLE customer_round_map AS
        WITH selected AS (
            SELECT c.c_customer_id, c.c_customer_sk, d.donor_customer_sk
            FROM customer_current_map c
            JOIN donor_customer_map d ON d.c_customer_id = c.c_customer_id
            ORDER BY RANDOM()
            LIMIT %s
        )
        SELECT
            c_customer_sk AS old_customer_sk,
            c_customer_id,
            donor_customer_sk,
            %s + ROW_NUMBER() OVER (ORDER BY c_customer_sk) AS new_customer_sk
        FROM selected
        """,
        (current_batch, max_customer_sk),
    )

    cur.execute("SELECT COUNT(*) FROM customer_round_map")
    selected_count = cur.fetchone()[0]
    if selected_count == 0:
        return 0, 0

    cur.execute(
        """
        UPDATE customer
        SET c_rec_end_date = CURRENT_DATE - 1
        WHERE c_customer_sk IN (SELECT old_customer_sk FROM customer_round_map)
        """
    )

    cur.execute(
        f"""
        INSERT INTO customer (
            c_customer_sk, c_customer_id, c_current_cdemo_sk, c_current_hdemo_sk,
            c_current_addr_sk, c_first_shipto_date_sk, c_first_sales_date_sk,
            c_salutation, c_first_name, c_last_name, c_preferred_cust_flag,
            c_birth_day, c_birth_month, c_birth_year, c_birth_country, c_login,
            c_email_address, {review_date_column}, c_rec_start_date, c_rec_end_date
        )
        SELECT
            map.new_customer_sk,
            src.c_customer_id,
            src.c_current_cdemo_sk,
            src.c_current_hdemo_sk,
            src.c_current_addr_sk,
            src.c_first_shipto_date_sk,
            src.c_first_sales_date_sk,
            src.c_salutation,
            src.c_first_name,
            src.c_last_name,
            src.c_preferred_cust_flag,
            src.c_birth_day,
            src.c_birth_month,
            src.c_birth_year,
            src.c_birth_country,
            src.c_login,
            src.c_email_address,
            src.{review_date_column},
            CURRENT_DATE,
            NULL
        FROM customer_round_map map
        JOIN customer src ON src.c_customer_sk = map.old_customer_sk
        ORDER BY map.new_customer_sk
        """
    )
    inserted_customers = cur.rowcount

    cur.execute(
        """
        UPDATE customer_current_map cur
        SET c_customer_sk = map.new_customer_sk
        FROM customer_round_map map
        WHERE cur.c_customer_id = map.c_customer_id
        """
    )

    cur.execute("SELECT COALESCE(MAX(ss_ticket_number), 0) FROM store_sales")
    max_ticket = cur.fetchone()[0]
    cur.execute(
        """
        WITH donor_rows AS (
            SELECT
                COALESCE(new_dd.d_date_sk, ss.ss_sold_date_sk) AS sold_date_sk,
                ss.ss_sold_time_sk,
                ss.ss_item_sk,
                map.new_customer_sk AS customer_sk,
                ss.ss_cdemo_sk,
                ss.ss_hdemo_sk,
                ss.ss_addr_sk,
                ss.ss_store_sk,
                ss.ss_promo_sk,
                ss.ss_quantity,
                ss.ss_wholesale_cost,
                ss.ss_list_price,
                ss.ss_sales_price,
                ss.ss_ext_discount_amt,
                ss.ss_ext_sales_price,
                ss.ss_ext_wholesale_cost,
                ss.ss_ext_list_price,
                ss.ss_ext_tax,
                ss.ss_coupon_amt,
                ss.ss_net_paid,
                ss.ss_net_paid_inc_tax,
                ss.ss_net_profit,
                ROW_NUMBER() OVER (ORDER BY map.new_customer_sk, ss.ss_ticket_number) AS rn
            FROM customer_round_map map
            JOIN LATERAL (
                SELECT *
                FROM donor_store_sales ss
                WHERE ss.ss_customer_sk = map.donor_customer_sk
                ORDER BY ss.ss_ticket_number
                LIMIT %s
            ) ss ON TRUE
            LEFT JOIN date_dim old_dd ON old_dd.d_date_sk = ss.ss_sold_date_sk
            LEFT JOIN date_dim new_dd ON new_dd.d_date = old_dd.d_date + %s
        )
        INSERT INTO store_sales (
            ss_sold_date_sk, ss_sold_time_sk, ss_item_sk, ss_customer_sk,
            ss_cdemo_sk, ss_hdemo_sk, ss_addr_sk, ss_store_sk, ss_promo_sk,
            ss_ticket_number, ss_quantity, ss_wholesale_cost, ss_list_price,
            ss_sales_price, ss_ext_discount_amt, ss_ext_sales_price,
            ss_ext_wholesale_cost, ss_ext_list_price, ss_ext_tax,
            ss_coupon_amt, ss_net_paid, ss_net_paid_inc_tax, ss_net_profit
        )
        SELECT
            sold_date_sk,
            ss_sold_time_sk,
            ss_item_sk,
            customer_sk,
            ss_cdemo_sk,
            ss_hdemo_sk,
            ss_addr_sk,
            ss_store_sk,
            ss_promo_sk,
            %s + rn,
            ss_quantity,
            ss_wholesale_cost,
            ss_list_price,
            ss_sales_price,
            ss_ext_discount_amt,
            ss_ext_sales_price,
            ss_ext_wholesale_cost,
            ss_ext_list_price,
            ss_ext_tax,
            ss_coupon_amt,
            ss_net_paid,
            ss_net_paid_inc_tax,
            ss_net_profit
        FROM donor_rows
        ORDER BY rn
        """,
        (facts_per_key, date_shift_days, max_ticket),
    )
    inserted_sales = cur.rowcount
    return inserted_customers, inserted_sales


def inject_scd2_drift(
    conn,
    rounds: int,
    drift_ratio: float,
    item_facts_per_key: int,
    customer_facts_per_key: int,
    date_shift_days: int,
) -> None:
    cur = conn.cursor()
    prepare_session_state(cur)
    conn.commit()

    try:
        for round_no in range(1, rounds + 1):
            print(f"\n--- Round {round_no}/{rounds} ---")

            inserted_items, item_sales = inject_item_round(
                cur,
                drift_ratio=drift_ratio,
                facts_per_key=item_facts_per_key,
                date_shift_days=date_shift_days,
            )
            print(
                f"  item: created {inserted_items} new keys, inserted {item_sales} fact rows."
            )

            inserted_customers, customer_sales = inject_customer_round(
                cur,
                drift_ratio=drift_ratio,
                facts_per_key=customer_facts_per_key,
                date_shift_days=date_shift_days,
            )
            print(
                f"  customer: created {inserted_customers} new keys, inserted {customer_sales} fact rows."
            )

            conn.commit()
    finally:
        cur.close()

    print("\nSCD2 drift injection complete.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dbname', default='tpcds')
    parser.add_argument('--user', default='postgres')
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--port', type=int, default=5433)
    parser.add_argument('--rounds', type=int, default=DRIFT_ROUNDS)
    parser.add_argument('--drift-ratio', type=float, default=DRIFT_RATIO)
    parser.add_argument('--item-facts-per-key', type=int, default=ITEM_FACTS_PER_KEY)
    parser.add_argument('--customer-facts-per-key', type=int, default=CUSTOMER_FACTS_PER_KEY)
    parser.add_argument('--date-shift-days', type=int, default=DATE_SHIFT_DAYS)
    parser.add_argument('--password', default='')
    args = parser.parse_args()

    conn = psycopg2.connect(
        host=args.host,
        port=args.port,
        dbname=args.dbname,
        user=args.user,
        password=args.password,
    )
    conn.autocommit = False

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
