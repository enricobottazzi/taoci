create extension if not exists citext;
create extension if not exists pgcrypto;

create table if not exists profiles (
  id             uuid primary key default gen_random_uuid(),
  username       citext unique not null check (char_length(username) between 2 and 32),
  password_hash  text not null,
  created_at     timestamptz not null default now()
);

create table if not exists submissions (
  submission_id    bigserial primary key,
  user_id          uuid not null references profiles(id) on delete cascade,
  feature_id       int  not null check (feature_id between 0 and 16383),
  label            text not null check (char_length(label) between 1 and 500),
  score            real not null check (score between 0 and 1),
  scorer_model_id  text not null,
  created_at       timestamptz not null default now()
);

create index if not exists submissions_feature_score_desc
  on submissions (feature_id, score desc, created_at asc);
create index if not exists submissions_user_idx
  on submissions (user_id);

create or replace view feature_best as
select distinct on (s.feature_id)
  s.feature_id,
  s.submission_id,
  s.user_id,
  p.username,
  s.label,
  s.score,
  s.scorer_model_id,
  s.created_at as found_at
from submissions s
join profiles p on p.id = s.user_id
order by s.feature_id, s.score desc, s.created_at asc;
