CREATE INDEX policy_scope_review IF NOT EXISTS FOR (n:Policy) ON (n.estate_id,n.review_status);
