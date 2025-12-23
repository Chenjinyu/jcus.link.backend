## Create Supabase
1. I created the supabase database via vercel which connect to Supabase.
2. run below sql to enable pgvector with postgres.
```sql
create extension if not exists vector;
```
