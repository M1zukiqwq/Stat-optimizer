-- 创建 IMDB 数据集的 Iceberg 表
-- JOB Benchmark 核心表（21 张表）

CREATE SCHEMA IF NOT EXISTS iceberg.imdb;

-- 1. title (电影/电视剧)
CREATE TABLE IF NOT EXISTS iceberg.imdb.title (
    id INTEGER,
    title VARCHAR,
    imdb_index VARCHAR,
    kind_id INTEGER,
    production_year INTEGER,
    imdb_id INTEGER,
    phonetic_code VARCHAR,
    episode_of_id INTEGER,
    season_nr INTEGER,
    episode_nr INTEGER,
    series_years VARCHAR,
    md5sum VARCHAR
) WITH (
    format = 'PARQUET',
    partitioning = ARRAY['production_year']
);

-- 2. cast_info (演员信息)
CREATE TABLE IF NOT EXISTS iceberg.imdb.cast_info (
    id INTEGER,
    person_id INTEGER,
    movie_id INTEGER,
    person_role_id INTEGER,
    note VARCHAR,
    nr_order INTEGER,
    role_id INTEGER
) WITH (format = 'PARQUET');

-- 3. movie_info (电影元信息)
CREATE TABLE IF NOT EXISTS iceberg.imdb.movie_info (
    id INTEGER,
    movie_id INTEGER,
    info_type_id INTEGER,
    info VARCHAR,
    note VARCHAR
) WITH (format = 'PARQUET');

-- 4. movie_companies (制片公司)
CREATE TABLE IF NOT EXISTS iceberg.imdb.movie_companies (
    id INTEGER,
    movie_id INTEGER,
    company_id INTEGER,
    company_type_id INTEGER,
    note VARCHAR
) WITH (format = 'PARQUET');

-- 5. movie_keyword (电影关键词)
CREATE TABLE IF NOT EXISTS iceberg.imdb.movie_keyword (
    id INTEGER,
    movie_id INTEGER,
    keyword_id INTEGER
) WITH (format = 'PARQUET');

-- 6. person_info (人物信息)
CREATE TABLE IF NOT EXISTS iceberg.imdb.person_info (
    id INTEGER,
    person_id INTEGER,
    info_type_id INTEGER,
    info VARCHAR,
    note VARCHAR
) WITH (format = 'PARQUET');

-- 7. movie_info_idx (电影索引信息)
CREATE TABLE IF NOT EXISTS iceberg.imdb.movie_info_idx (
    id INTEGER,
    movie_id INTEGER,
    info_type_id INTEGER,
    info VARCHAR,
    note VARCHAR
) WITH (format = 'PARQUET');

-- 8. aka_title (别名)
CREATE TABLE IF NOT EXISTS iceberg.imdb.aka_title (
    id INTEGER,
    movie_id INTEGER,
    title VARCHAR,
    imdb_index VARCHAR,
    kind_id INTEGER,
    production_year INTEGER,
    phonetic_code VARCHAR,
    episode_of_id INTEGER,
    season_nr INTEGER,
    episode_nr INTEGER,
    note VARCHAR,
    md5sum VARCHAR
) WITH (format = 'PARQUET');

-- 9. aka_name (人名别名)
CREATE TABLE IF NOT EXISTS iceberg.imdb.aka_name (
    id INTEGER,
    person_id INTEGER,
    name VARCHAR,
    imdb_index VARCHAR,
    name_pcode_cf VARCHAR,
    name_pcode_nf VARCHAR,
    surname_pcode VARCHAR,
    md5sum VARCHAR
) WITH (format = 'PARQUET');

-- 10. complete_cast (完整演员表)
CREATE TABLE IF NOT EXISTS iceberg.imdb.complete_cast (
    id INTEGER,
    movie_id INTEGER,
    subject_id INTEGER,
    status_id INTEGER
) WITH (format = 'PARQUET');

-- 11-21. 维度表（小表）
CREATE TABLE IF NOT EXISTS iceberg.imdb.comp_cast_type (
    id INTEGER,
    kind VARCHAR
) WITH (format = 'PARQUET');

CREATE TABLE IF NOT EXISTS iceberg.imdb.company_name (
    id INTEGER,
    name VARCHAR,
    country_code VARCHAR,
    imdb_id INTEGER,
    name_pcode_nf VARCHAR,
    name_pcode_sf VARCHAR,
    md5sum VARCHAR
) WITH (format = 'PARQUET');

CREATE TABLE IF NOT EXISTS iceberg.imdb.company_type (
    id INTEGER,
    kind VARCHAR
) WITH (format = 'PARQUET');

CREATE TABLE IF NOT EXISTS iceberg.imdb.info_type (
    id INTEGER,
    info VARCHAR
) WITH (format = 'PARQUET');

CREATE TABLE IF NOT EXISTS iceberg.imdb.keyword (
    id INTEGER,
    keyword VARCHAR,
    phonetic_code VARCHAR
) WITH (format = 'PARQUET');

CREATE TABLE IF NOT EXISTS iceberg.imdb.kind_type (
    id INTEGER,
    kind VARCHAR
) WITH (format = 'PARQUET');

CREATE TABLE IF NOT EXISTS iceberg.imdb.link_type (
    id INTEGER,
    link VARCHAR
) WITH (format = 'PARQUET');

CREATE TABLE IF NOT EXISTS iceberg.imdb.name (
    id INTEGER,
    name VARCHAR,
    imdb_index VARCHAR,
    imdb_id INTEGER,
    gender VARCHAR,
    name_pcode_cf VARCHAR,
    name_pcode_nf VARCHAR,
    surname_pcode VARCHAR,
    md5sum VARCHAR
) WITH (format = 'PARQUET');

CREATE TABLE IF NOT EXISTS iceberg.imdb.role_type (
    id INTEGER,
    role VARCHAR
) WITH (format = 'PARQUET');

CREATE TABLE IF NOT EXISTS iceberg.imdb.char_name (
    id INTEGER,
    name VARCHAR,
    imdb_index VARCHAR,
    imdb_id INTEGER,
    name_pcode_nf VARCHAR,
    surname_pcode VARCHAR,
    md5sum VARCHAR
) WITH (format = 'PARQUET');

CREATE TABLE IF NOT EXISTS iceberg.imdb.movie_link (
    id INTEGER,
    movie_id INTEGER,
    linked_movie_id INTEGER,
    link_type_id INTEGER
) WITH (format = 'PARQUET');
