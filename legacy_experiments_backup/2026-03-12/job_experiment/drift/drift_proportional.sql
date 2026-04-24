-- 按比例漂移注入 - 每轮影响表大小的 2%
-- 分配: INSERT 0.8%, DELETE 0.6%, UPDATE 0.6%

-- title 表 (原约 252万行)
-- INSERT ~2万行 (复制并修改)
INSERT INTO iceberg.imdb.title 
SELECT 
    id + 100000000,
    title || '_copy',
    imdb_index,
    kind_id,
    production_year + 1,
    imdb_id,
    phonetic_code,
    episode_of_id,
    season_nr,
    episode_nr,
    series_years,
    md5sum
FROM iceberg.imdb.title
WHERE id % 100 < 1
LIMIT 20000;

-- cast_info 表 (原约 3624万行)
INSERT INTO iceberg.imdb.cast_info 
SELECT 
    id + 1000000000,
    person_id,
    movie_id + 100000000,
    person_role_id,
    note,
    nr_order,
    role_id
FROM iceberg.imdb.cast_info
WHERE id % 100 < 1
LIMIT 300000;

-- movie_info 表 (原约 1483万行)
INSERT INTO iceberg.imdb.movie_info 
SELECT 
    id + 1000000000,
    movie_id + 100000000,
    info_type_id,
    info || '_copy',
    note
FROM iceberg.imdb.movie_info
WHERE id % 100 < 1
LIMIT 120000;

-- movie_companies 表 (原约 260万行)
INSERT INTO iceberg.imdb.movie_companies 
SELECT 
    id + 1000000000,
    movie_id + 100000000,
    company_id,
    company_type_id,
    note
FROM iceberg.imdb.movie_companies
WHERE id % 100 < 1
LIMIT 20000;

-- name 表 (原约 416万行)
INSERT INTO iceberg.imdb.name 
SELECT 
    id + 1000000000,
    name || '_copy',
    imdb_index,
    imdb_id,
    gender,
    name_pcode_cf,
    name_pcode_nf,
    surname_pcode,
    md5sum
FROM iceberg.imdb.name
WHERE id % 100 < 1
LIMIT 30000;

-- movie_keyword 表 (原约 452万行)
INSERT INTO iceberg.imdb.movie_keyword 
SELECT 
    id + 1000000000,
    movie_id + 100000000,
    keyword_id
FROM iceberg.imdb.movie_keyword
WHERE id % 100 < 1
LIMIT 35000;

-- 执行 DELETE（删除刚才插入的部分数据，模拟更新）
DELETE FROM iceberg.imdb.title WHERE id > 100000000 AND id % 3 = 0;
DELETE FROM iceberg.imdb.cast_info WHERE id > 1000000000 AND id % 3 = 0;
DELETE FROM iceberg.imdb.movie_info WHERE id > 1000000000 AND id % 3 = 0;
DELETE FROM iceberg.imdb.movie_companies WHERE id > 1000000000 AND id % 3 = 0;
DELETE FROM iceberg.imdb.name WHERE id > 1000000000 AND id % 3 = 0;
DELETE FROM iceberg.imdb.movie_keyword WHERE id > 1000000000 AND id % 3 = 0;
