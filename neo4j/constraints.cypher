// DDL only; all runtime data access additionally requires estate_id in fixed Cypher.
CREATE CONSTRAINT estate_v2_id IF NOT EXISTS FOR (n:Estate) REQUIRE n.estate_id IS UNIQUE;
CREATE CONSTRAINT building_v2_id IF NOT EXISTS FOR (n:Building) REQUIRE n.building_id IS UNIQUE;
CREATE CONSTRAINT unit_v2_id IF NOT EXISTS FOR (n:Unit) REQUIRE n.unit_id IS UNIQUE;
CREATE CONSTRAINT resident_v2_id IF NOT EXISTS FOR (n:Resident) REQUIRE n.resident_id IS UNIQUE;
CREATE CONSTRAINT service_v2_id IF NOT EXISTS FOR (n:Service) REQUIRE n.service_id IS UNIQUE;
CREATE CONSTRAINT policy_v2_id IF NOT EXISTS FOR (n:Policy) REQUIRE n.policy_id IS UNIQUE;
CREATE CONSTRAINT document_v2_id IF NOT EXISTS FOR (n:Document) REQUIRE n.document_id IS UNIQUE;
CREATE CONSTRAINT evidencechunk_v2_id IF NOT EXISTS FOR (n:EvidenceChunk) REQUIRE n.chunk_id IS UNIQUE;
CREATE CONSTRAINT workorder_v2_id IF NOT EXISTS FOR (n:WorkOrder) REQUIRE n.work_order_id IS UNIQUE;
CREATE CONSTRAINT reviewdecision_v2_id IF NOT EXISTS FOR (n:ReviewDecision) REQUIRE n.review_decision_id IS UNIQUE;
CREATE CONSTRAINT service_scope_code IF NOT EXISTS FOR (n:Service) REQUIRE (n.estate_id,n.service_code) IS UNIQUE;
CREATE CONSTRAINT work_order_scope_number IF NOT EXISTS FOR (n:WorkOrder) REQUIRE (n.estate_id,n.number) IS UNIQUE;
CREATE CONSTRAINT work_order_scope_request IF NOT EXISTS FOR (n:WorkOrder) REQUIRE (n.estate_id,n.resident_id,n.idempotency_key) IS UNIQUE;
