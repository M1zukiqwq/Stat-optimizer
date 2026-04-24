-- 改进的数据漂移脚本
-- 使用动态 ID 生成，避免冲突
-- 参数: {ROUND} 将被脚本替换为当前轮次

-- 1. 插入新数据（2% 增量，使用轮次相关的随机偏移）
INSERT INTO iceberg.imdb.title
SELECT
    id + (20000000 * {ROUND}) + CAST(RAND() * 1000000 AS BIGINT),
    title || '_drift_r{ROUND}',
    imdb_index,
    kind_id,
    production_year + {ROUND},
    imdb_id,
    phonetic_code,
    episode_of_id,
    season_nr,
    episode_nr,
    series_years,
    md5sum
FROM iceberg.imdb.title
WHERE MOD(CAST(RAND() * 1000000 AS BIGINT), 50) = 0
LIMIT (SELECT CAST(COUNT(*) * 0.02 AS BIGINT) FROM iceberg.imdb.title);

INSERT INTO iceberg.imdb.name
SELECT
    id + (20000000 * {ROUND}) + CAST(RAND() * 1000000 AS BIGINT),
    name || '_drift_r{ROUND}',
    imdb_index,
    imdb_id,
    gender,
    name_pcode_cf,
    name_pcode_nf,
    surname_pcode,
    md5sum
FROM iceberg.imdb.name
WHERE MOD(CAST(RAND() * 1000000 AS BIGINT), 50) = 0
LIMIT (SELECT CAST(COUNT(*) * 0.02 AS BIGINT) FROM iceberg.imdb.name);

-- 2. 插入关联表数据（保持外键一致性）
INSERT INTO iceberg.imdb.cast_info
SELECT
    id + (200000000 * {ROUND}) + CAST(RAND() * 1000000 AS BIGINT),
    person_id,
    movie_id,
    person_role_id,
    note || '_drift_r{ROUND}',
    nr_order,
    role_id
FROM iceberg.imdb.cast_info
WHERE MOD(CAST(RAND() * 1000000 AS BIGINT), 50) = 0
LIMIT (SELECT CAST(COUNT(*) * 0.02 AS BIGINT) FROM iceberg.imdb.cast_info);

INSERT INTO iceberg.imdb.movie_info
SELECT
    id + (200000000 * {ROUND}) + CAST(RAND() * 1000000 AS BIGINT),
    movie_id,
    info_type_id,
    info || '_drift_r{ROUND}',
    note
FROM iceberg.imdb.movie_info
WHERE MOD(CAST(RAND() * 1000000 AS BIGINT), 50) = 0
LIMIT (SELECT CAST(COUNT(*) * 0.02 AS BIGINT) FROM iceberg.imdb.movie_info);

INSERT INTO iceberg.imdb.movie_companies
SELECT
    id + (200000000 * {ROUND}) + CAST(RAND() * 1000000 AS BIGINT),
    movie_id,
    company_id,
    company_type_id,
    note || '_drift_r{ROUND}'
FROM iceberg.imdb.movie_companies
WHERE MOD(CAST(RAND() * 1000000 AS BIGINT), 50) = 0
LIMIT (SELECT CAST(COUNT(*) * 0.02 AS BIGINT) FROM iceberg.imdb.movie_companies);

INSERT INTO iceberg.imdb.movie_keyword
SELECT
    id + (200000000 * {ROUND}) + CAST(RAND() * 1000000 AS BIGINT),
    movie_id,
    keyword_id
FROM iceberg.imdb.movie_keyword
WHERE MOD(CAST(RAND() * 1000000 AS BIGINT), 50) = 0
LIMIT (SELECT CAST(COUNT(*) * 0.02 AS BIGINT) FROM iceberg.imdb.movie_keyword);

-- 3. 删除旧数据（1% 随机删除）
DELETE FROM iceberg.imdb.title
WHERE id IN (
    SELECT id FROM iceberg.imdb.title
    WHERE MOD(CAST(RAND() * 1000000 AS BIGINT), 100) = {ROUND}
    LIMIT (SELECT CAST(COUNT(*) * 0.01 AS BIGINT) FROM iceberg.imdb.title)
);

DELETE FROM iceberg.imdb.cast_info
WHERE id IN (
    SELECT id FROM iceberg.imdb.cast_info
    WHERE MOD(CAST(RAND() * 1000000 AS BIGINT), 100) = {ROUND}
    LIMIT (SELECT CAST(COUNT(*) * 0.01 AS BIGINT) FROM iceberg.imdb.cast_info)
);

DELETE FROM iceberg.imdb.movie_info
WHERE id IN (
    SELECT id FROM iceberg.imdb.movie_info
    WHERE MOD(CAST(RAND() * 1000000 AS BIGINT), 100) = {ROUND}
    LIMIT (SELECT CAST(COUNT(*) * 0.01 AS BIGINT) FROM iceberg.imdb.movie_info)
);

DELETE FROM iceberg.imdb.movie_companies
WHERE id IN (
    SELECT id FROM iceberg.imdb.movie_companies
    WHERE MOD(CAST(RAND() * 1000000 AS BIGINT), 100) = {ROUND}
    LIMIT (SELECT CAST(COUNT(*) * 0.01 AS BIGINT) FROM iceberg.imdb.movie_companies)
);

DELETE FROM iceberg.imdb.name
WHERE id IN (
    SELECT id FROM iceberg.imdb.name
    WHERE MOD(CAST(RAND() * 1000000 AS BIGINT), 100) = {ROUND}
    LIMIT (SELECT CAST(COUNT(*) * 0.01 AS BIGINT) FROM iceberg.imdb.name)
);

DELETE FROM iceberg.imdb.movie_keyword
WHERE id IN (
    SELECT id FROM iceberg.imdb.movie_keyword
    WHERE MOD(CAST(RAND() * 1000000 AS BIGINT), 100) = {ROUND}
    LIMIT (SELECT CAST(COUNT(*) * 0.01 AS BIGINT) FROM iceberg.imdb.movie_keyword)
);
