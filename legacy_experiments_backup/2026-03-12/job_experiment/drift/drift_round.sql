-- 漂移注入脚本 - Round {round_num}
-- 影响 title, cast_info, movie_info, movie_companies, name, movie_keyword 表

-- 1. title 表: INSERT 5万行副本（修改ID和年份）
INSERT INTO iceberg.imdb.title 
SELECT 
    id + {round_num} * 10000000,
    title || '_drift_{round_num}',
    imdb_index,
    kind_id,
    production_year + {round_num},
    imdb_id,
    phonetic_code,
    episode_of_id,
    season_nr,
    episode_nr,
    series_years,
    md5sum
FROM iceberg.imdb.title
WHERE id % 20 = {round_num} % 20
LIMIT 50000;

-- 2. cast_info 表: INSERT 100万行副本
INSERT INTO iceberg.imdb.cast_info 
SELECT 
    id + {round_num} * 100000000,
    person_id,
    movie_id + {round_num} * 10000000,
    person_role_id,
    note,
    nr_order,
    role_id
FROM iceberg.imdb.cast_info
WHERE id % 36 = {round_num} % 36
LIMIT 1000000;

-- 3. movie_info 表: INSERT 50万行副本
INSERT INTO iceberg.imdb.movie_info 
SELECT 
    id + {round_num} * 100000000,
    movie_id + {round_num} * 10000000,
    info_type_id,
    info || '_drift_{round_num}',
    note
FROM iceberg.imdb.movie_info
WHERE id % 30 = {round_num} % 30
LIMIT 500000;

-- 4. movie_companies 表: INSERT 10万行副本
INSERT INTO iceberg.imdb.movie_companies 
SELECT 
    id + {round_num} * 100000000,
    movie_id + {round_num} * 10000000,
    company_id,
    company_type_id,
    note
FROM iceberg.imdb.movie_companies
WHERE id % 26 = {round_num} % 26
LIMIT 100000;

-- 5. name 表: INSERT 20万行副本
INSERT INTO iceberg.imdb.name 
SELECT 
    id + {round_num} * 100000000,
    name || '_drift_{round_num}',
    imdb_index,
    imdb_id,
    gender,
    name_pcode_cf,
    name_pcode_nf,
    surname_pcode,
    md5sum
FROM iceberg.imdb.name
WHERE id % 20 = {round_num} % 20
LIMIT 200000;

-- 6. movie_keyword 表: INSERT 20万行副本
INSERT INTO iceberg.imdb.movie_keyword 
SELECT 
    id + {round_num} * 100000000,
    movie_id + {round_num} * 10000000,
    keyword_id
FROM iceberg.imdb.movie_keyword
WHERE id % 22 = {round_num} % 22
LIMIT 200000;
