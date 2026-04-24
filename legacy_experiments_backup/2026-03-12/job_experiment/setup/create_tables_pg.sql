-- PostgreSQL IMDB 表结构
-- 用于 JOB Benchmark 端到端实验

-- 1. title (电影/电视剧)
CREATE TABLE IF NOT EXISTS title (
    id INTEGER PRIMARY KEY,
    title VARCHAR(1000),
    imdb_index VARCHAR(12),
    kind_id INTEGER,
    production_year INTEGER,
    imdb_id INTEGER,
    phonetic_code VARCHAR(5),
    episode_of_id INTEGER,
    season_nr INTEGER,
    episode_nr INTEGER,
    series_years VARCHAR(49),
    md5sum VARCHAR(32)
);

-- 2. cast_info (演员信息)
CREATE TABLE IF NOT EXISTS cast_info (
    id INTEGER PRIMARY KEY,
    person_id INTEGER,
    movie_id INTEGER,
    person_role_id INTEGER,
    note TEXT,
    nr_order INTEGER,
    role_id INTEGER
);

-- 3. movie_info (电影元信息)
CREATE TABLE IF NOT EXISTS movie_info (
    id INTEGER PRIMARY KEY,
    movie_id INTEGER,
    info_type_id INTEGER,
    info TEXT,
    note TEXT
);

-- 4. movie_companies (制片公司)
CREATE TABLE IF NOT EXISTS movie_companies (
    id INTEGER PRIMARY KEY,
    movie_id INTEGER,
    company_id INTEGER,
    company_type_id INTEGER,
    note TEXT
);

-- 5. movie_keyword (电影关键词)
CREATE TABLE IF NOT EXISTS movie_keyword (
    id INTEGER PRIMARY KEY,
    movie_id INTEGER,
    keyword_id INTEGER
);

-- 6. person_info (人物信息)
CREATE TABLE IF NOT EXISTS person_info (
    id INTEGER PRIMARY KEY,
    person_id INTEGER,
    info_type_id INTEGER,
    info TEXT,
    note TEXT
);

-- 7. movie_info_idx (电影索引信息)
CREATE TABLE IF NOT EXISTS movie_info_idx (
    id INTEGER PRIMARY KEY,
    movie_id INTEGER,
    info_type_id INTEGER,
    info VARCHAR(1000),
    note TEXT
);

-- 8. aka_title (别名)
CREATE TABLE IF NOT EXISTS aka_title (
    id INTEGER PRIMARY KEY,
    movie_id INTEGER,
    title VARCHAR(1000),
    imdb_index VARCHAR(12),
    kind_id INTEGER,
    production_year INTEGER,
    phonetic_code VARCHAR(5),
    episode_of_id INTEGER,
    season_nr INTEGER,
    episode_nr INTEGER,
    note TEXT,
    md5sum VARCHAR(32)
);

-- 9. aka_name (人名别名)
CREATE TABLE IF NOT EXISTS aka_name (
    id INTEGER PRIMARY KEY,
    person_id INTEGER,
    name VARCHAR(1000),
    imdb_index VARCHAR(12),
    name_pcode_cf VARCHAR(6),
    name_pcode_nf VARCHAR(9),
    surname_pcode VARCHAR(6),
    md5sum VARCHAR(32)
);

-- 10. complete_cast (完整演员表)
CREATE TABLE IF NOT EXISTS complete_cast (
    id INTEGER PRIMARY KEY,
    movie_id INTEGER,
    subject_id INTEGER,
    status_id INTEGER
);

-- 11. comp_cast_type
CREATE TABLE IF NOT EXISTS comp_cast_type (
    id INTEGER PRIMARY KEY,
    kind VARCHAR(32)
);

-- 12. company_name
CREATE TABLE IF NOT EXISTS company_name (
    id INTEGER PRIMARY KEY,
    name TEXT,
    country_code VARCHAR(6),
    imdb_id INTEGER,
    name_pcode_nf VARCHAR(6),
    name_pcode_sf VARCHAR(6),
    md5sum VARCHAR(32)
);

-- 13. company_type
CREATE TABLE IF NOT EXISTS company_type (
    id INTEGER PRIMARY KEY,
    kind VARCHAR(32)
);

-- 14. info_type
CREATE TABLE IF NOT EXISTS info_type (
    id INTEGER PRIMARY KEY,
    info VARCHAR(1000)
);

-- 15. keyword
CREATE TABLE IF NOT EXISTS keyword (
    id INTEGER PRIMARY KEY,
    keyword VARCHAR(1000),
    phonetic_code VARCHAR(5)
);

-- 16. kind_type
CREATE TABLE IF NOT EXISTS kind_type (
    id INTEGER PRIMARY KEY,
    kind VARCHAR(15)
);

-- 17. link_type
CREATE TABLE IF NOT EXISTS link_type (
    id INTEGER PRIMARY KEY,
    link VARCHAR(32)
);

-- 18. name (人物)
CREATE TABLE IF NOT EXISTS name (
    id INTEGER PRIMARY KEY,
    name VARCHAR(1000),
    imdb_index VARCHAR(12),
    imdb_id INTEGER,
    gender VARCHAR(1),
    name_pcode_cf VARCHAR(6),
    name_pcode_nf VARCHAR(9),
    surname_pcode VARCHAR(6),
    md5sum VARCHAR(32)
);

-- 19. role_type
CREATE TABLE IF NOT EXISTS role_type (
    id INTEGER PRIMARY KEY,
    role VARCHAR(32)
);

-- 20. char_name (角色名)
CREATE TABLE IF NOT EXISTS char_name (
    id INTEGER PRIMARY KEY,
    name TEXT,
    imdb_index VARCHAR(12),
    imdb_id INTEGER,
    name_pcode_nf VARCHAR(6),
    surname_pcode VARCHAR(6),
    md5sum VARCHAR(32)
);

-- 21. movie_link
CREATE TABLE IF NOT EXISTS movie_link (
    id INTEGER PRIMARY KEY,
    movie_id INTEGER,
    linked_movie_id INTEGER,
    link_type_id INTEGER
);
