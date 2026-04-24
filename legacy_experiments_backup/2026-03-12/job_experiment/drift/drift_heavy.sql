-- 重度漂移：每轮 INSERT 2% + DELETE 1% + UPDATE 1%
-- 共 15 轮，累计影响约 30%

-- INSERT 新数据（模拟新数据写入）
INSERT INTO iceberg.imdb.title 
SELECT id + 20000000, title || '_drift', imdb_index, kind_id, production_year + 1, imdb_id, phonetic_code, episode_of_id, season_nr, episode_nr, series_years, md5sum 
FROM iceberg.imdb.title WHERE id % 50 = 0;

INSERT INTO iceberg.imdb.cast_info 
SELECT id + 200000000, person_id, movie_id + 20000000, person_role_id, note, nr_order, role_id 
FROM iceberg.imdb.cast_info WHERE id % 50 = 0;

INSERT INTO iceberg.imdb.movie_info 
SELECT id + 200000000, movie_id + 20000000, info_type_id, info || '_drift', note 
FROM iceberg.imdb.movie_info WHERE id % 50 = 0;

INSERT INTO iceberg.imdb.movie_companies 
SELECT id + 200000000, movie_id + 20000000, company_id, company_type_id, note 
FROM iceberg.imdb.movie_companies WHERE id % 50 = 0;

INSERT INTO iceberg.imdb.name 
SELECT id + 200000000, name || '_drift', imdb_index, imdb_id, gender, name_pcode_cf, name_pcode_nf, surname_pcode, md5sum 
FROM iceberg.imdb.name WHERE id % 50 = 0;

INSERT INTO iceberg.imdb.movie_keyword 
SELECT id + 200000000, movie_id + 20000000, keyword_id 
FROM iceberg.imdb.movie_keyword WHERE id % 50 = 0;

-- DELETE 部分旧数据（模拟数据过期）
DELETE FROM iceberg.imdb.title WHERE id % 100 = 1;
DELETE FROM iceberg.imdb.cast_info WHERE id % 100 = 1;
DELETE FROM iceberg.imdb.movie_info WHERE id % 100 = 1;
DELETE FROM iceberg.imdb.movie_companies WHERE id % 100 = 1;
DELETE FROM iceberg.imdb.name WHERE id % 100 = 1;
DELETE FROM iceberg.imdb.movie_keyword WHERE id % 100 = 1;
