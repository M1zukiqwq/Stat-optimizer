CREATE TABLE aka_name (
    id integer,
    person_id integer,
    name text,
    imdb_index character varying(12),
    name_pcode_cf character varying(5),
    name_pcode_nf character varying(5),
    surname_pcode character varying(5),
    md5sum character varying(32)
);

CREATE TABLE aka_title (
    id integer,
    movie_id integer,
    title text,
    imdb_index character varying(12),
    kind_id integer,
    production_year integer,
    phonetic_code character varying(5),
    episode_of_id integer,
    season_nr integer,
    episode_nr integer,
    note text,
    md5sum character varying(32)
);

CREATE TABLE cast_info (
    id integer,
    person_id integer,
    movie_id integer,
    person_role_id integer,
    note text,
    nr_order integer,
    role_id integer
);

CREATE TABLE char_name (
    id integer,
    name text,
    imdb_index character varying(12),
    imdb_id text,
    name_pcode_nf character varying(5),
    surname_pcode character varying(5),
    md5sum character varying(32)
);

CREATE TABLE comp_cast_type (
    id integer,
    kind character varying(32)
);

CREATE TABLE company_name (
    id integer,
    name text,
    country_code character varying(255),
    imdb_id text,
    name_pcode_nf character varying(5),
    name_pcode_sf character varying(5),
    md5sum character varying(32)
);

CREATE TABLE company_type (
    id integer,
    kind character varying(32)
);

CREATE TABLE complete_cast (
    id integer,
    movie_id integer,
    subject_id integer,
    status_id integer
);

CREATE TABLE info_type (
    id integer,
    info character varying(32)
);

CREATE TABLE keyword (
    id integer,
    keyword text,
    phonetic_code character varying(5)
);

CREATE TABLE kind_type (
    id integer,
    kind character varying(15)
);

CREATE TABLE link_type (
    id integer,
    link character varying(32)
);

CREATE TABLE movie_companies (
    id integer,
    movie_id integer,
    company_id integer,
    company_type_id integer,
    note text
);

CREATE TABLE movie_info (
    id integer,
    movie_id integer,
    info_type_id integer,
    info text,
    note text
);

CREATE TABLE movie_info_idx (
    id integer,
    movie_id integer,
    info_type_id integer,
    info text,
    note text
);

CREATE TABLE movie_keyword (
    id integer,
    movie_id integer,
    keyword_id integer
);

CREATE TABLE movie_link (
    id integer,
    movie_id integer,
    linked_movie_id integer,
    link_type_id integer
);

CREATE TABLE name (
    id integer,
    name text,
    imdb_index character varying(12),
    imdb_id text,
    gender character varying(1),
    name_pcode_cf character varying(5),
    name_pcode_nf character varying(5),
    surname_pcode character varying(5),
    md5sum character varying(32)
);

CREATE TABLE person_info (
    id integer,
    person_id integer,
    info_type_id integer,
    info text,
    note text
);

CREATE TABLE role_type (
    id integer,
    role character varying(32)
);

CREATE TABLE title (
    id integer,
    title text,
    imdb_index character varying(12),
    kind_id integer,
    production_year integer,
    imdb_id text,
    phonetic_code character varying(5),
    episode_of_id integer,
    season_nr integer,
    episode_nr integer,
    series_years character varying(49),
    md5sum character varying(32)
);
