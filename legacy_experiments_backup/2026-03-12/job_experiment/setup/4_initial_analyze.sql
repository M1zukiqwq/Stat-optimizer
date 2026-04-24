-- 对所有 IMDB 表运行 ANALYZE，生成 KLL Sketch 统计

ANALYZE iceberg.imdb.title;
ANALYZE iceberg.imdb.cast_info;
ANALYZE iceberg.imdb.movie_info;
ANALYZE iceberg.imdb.movie_companies;
ANALYZE iceberg.imdb.movie_keyword;
ANALYZE iceberg.imdb.person_info;
ANALYZE iceberg.imdb.movie_info_idx;
ANALYZE iceberg.imdb.aka_title;
ANALYZE iceberg.imdb.aka_name;
ANALYZE iceberg.imdb.complete_cast;
ANALYZE iceberg.imdb.comp_cast_type;
ANALYZE iceberg.imdb.company_name;
ANALYZE iceberg.imdb.company_type;
ANALYZE iceberg.imdb.info_type;
ANALYZE iceberg.imdb.keyword;
ANALYZE iceberg.imdb.kind_type;
ANALYZE iceberg.imdb.link_type;
ANALYZE iceberg.imdb.name;
ANALYZE iceberg.imdb.role_type;
ANALYZE iceberg.imdb.char_name;
ANALYZE iceberg.imdb.movie_link;
