CREATE TABLE IF NOT EXISTS articles (
    article_id TEXT PRIMARY KEY,
    title TEXT,
    content TEXT,
    source TEXT,
    published_at TIMESTAMP,
    url TEXT
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    article_id TEXT REFERENCES articles(article_id),
    content TEXT,
    chunk_index INTEGER
);
