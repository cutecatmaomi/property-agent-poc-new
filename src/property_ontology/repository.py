"""Private, fixed Cypher operations. Callers cannot supply a query, label or estate expression."""

import json
import uuid

from neo4j import GraphDatabase

from .domain import CORE_IDS, RELATIONS, TRANSITIONS, POCError, association_valid, canonical, estate

AVAILABLE = "资源不可用或无权访问"


class GraphRepository:
    def __init__(self, settings):
        self.settings = settings
        self.driver = GraphDatabase.driver(
            settings.uri,
            auth=(settings.username, settings.password),
            connection_timeout=5,
            max_transaction_retry_time=10,
        )

    def close(self):
        self.driver.close()

    def _read(self, scope, function):
        estate(scope)
        with self.driver.session(database=self.settings.database) as session:
            return session.execute_read(function)

    def _write(self, scope, function):
        estate(scope)
        with self.driver.session(database=self.settings.database) as session:
            return session.execute_write(function)

    def initialize_schema(self):
        # DDL is the sole scope-independent operation; it never reads/writes tenant records.
        with self.driver.session(database=self.settings.database) as session:
            for label, key in CORE_IDS.items():
                session.run(
                    f"CREATE CONSTRAINT {label.lower()}_v2_id IF NOT EXISTS "
                    f"FOR (n:{label}) REQUIRE n.{key} IS UNIQUE"
                ).consume()
            for name, label, keys in [
                ("service_scope_code", "Service", "n.estate_id,n.service_code"),
                ("work_order_scope_number", "WorkOrder", "n.estate_id,n.number"),
                ("work_order_scope_request", "WorkOrder", "n.estate_id,n.resident_id,n.idempotency_key"),
            ]:
                session.run(
                    f"CREATE CONSTRAINT {name} IF NOT EXISTS FOR (n:{label}) REQUIRE ({keys}) IS UNIQUE"
                ).consume()
            session.run(
                "CREATE INDEX policy_scope_review IF NOT EXISTS FOR (n:Policy) "
                "ON (n.estate_id,n.review_status)"
            ).consume()
            session.run(
                "CREATE INDEX faq_scope_review IF NOT EXISTS FOR (n:FAQ) ON (n.estate_id,n.review_status)"
            ).consume()

    def seed(self, scope, nodes, edges):
        estate(scope)
        for label, rows in nodes.items():
            if label not in CORE_IDS:
                raise ValueError("非受控对象")
            key = CORE_IDS[label]
            for row in rows:
                if row.get("estate_id") != scope or not (
                    row[key] == scope or row[key].startswith(scope + ":")
                ):
                    raise ValueError("导入节点跨楼盘")
        for edge in edges:
            expected = RELATIONS.get(edge["relation"])
            if (
                not expected
                or not (edge["source_type"] in expected[0] if isinstance(expected[0], set)
                        else expected[0] == edge["source_type"])
                or not (
                    edge["target_type"] in expected[1]
                    if isinstance(expected[1], set)
                    else edge["target_type"] == expected[1]
                )
            ):
                raise ValueError("非受控关系")
            if edge["estate_id"] != scope or any(
                not (edge[k] == scope or edge[k].startswith(scope + ":")) for k in ["source_id", "target_id"]
            ):
                raise ValueError("导入关系跨楼盘")
            if edge.get("properties", {}).get("estate_id", scope) != scope:
                raise ValueError("关系属性不能覆盖楼盘")

        def seed_tx(tx):
            for label, rows in nodes.items():
                key = CORE_IDS[label]
                for row in rows:
                    result = tx.run(
                        f"MERGE (n:{label} {{estate_id:$estate_id,{key}:$identifier}}) "
                        "ON CREATE SET n += $props RETURN n._record_json AS stored",
                        estate_id=scope,
                        identifier=row[key],
                        props=row,
                    ).single()
                    if row.get("_record_json") and result["stored"] != row["_record_json"]:
                        raise ValueError("同一不可变ID对应不同内容，停止导入")
            for edge in edges:
                src, dst = edge["source_type"], edge["target_type"]
                rel = edge["relation"]
                result = tx.run(
                    f"MATCH (a:{src} {{estate_id:$estate_id,{CORE_IDS[src]}:$source_id}}), "
                    f"(b:{dst} {{estate_id:$estate_id,{CORE_IDS[dst]}:$target_id}}) "
                    f"MERGE (a)-[r:{rel} {{estate_id:$estate_id}}]->(b) "
                    "ON CREATE SET r += $props RETURN count(r) AS count",
                    estate_id=scope,
                    source_id=edge["source_id"],
                    target_id=edge["target_id"],
                    props={"estate_id": scope, **edge.get("properties", {})},
                ).single()
                if result["count"] != 1:
                    raise ValueError("关系端点丢失或重复，回滚当前楼盘")

        self._write(scope, seed_tx)

    @staticmethod
    def _eligible(tx, scope, resident, unit, day, create, service_code="maintenance"):
        result = tx.run(
            """
            MATCH (e:Estate {estate_id:$estate_id})-[eb:HAS_BUILDING {estate_id:$estate_id}]->
                  (b:Building {estate_id:$estate_id})-[bu:HAS_UNIT {estate_id:$estate_id}]->
                  (u:Unit {estate_id:$estate_id,unit_id:$unit_id})
            MATCH (r:Resident {estate_id:$estate_id,resident_id:$resident_id,status:'ACTIVE'})-
                  [a:ASSOCIATED_WITH {estate_id:$estate_id}]->(u)
            MATCH (e)-[:OFFERS {estate_id:$estate_id}]->
                  (s:Service {estate_id:$estate_id,service_code:$service_code,enabled:true})
            WHERE u.building_id=b.building_id AND u.status='ACTIVE'
            RETURN properties(a) AS association, properties(u) AS unit, properties(b) AS building,
                   s.service_id AS service_id
            """,
            estate_id=scope,
            resident_id=resident,
            unit_id=unit,
            service_code=service_code,
        ).data()
        for row in result:
            if association_valid(row["association"], day, create=create):
                return row
        raise POCError("RESOURCE_NOT_AVAILABLE", AVAILABLE, 404)

    def eligible_unit(self, who, unit_id, day, create=True):
        return self._read(
            who.estate_id, lambda tx: self._eligible(tx, who.estate_id, who.actor_ref, unit_id, day, create)
        )

    def resident_active(self, scope, resident_id):
        return self._read(
            scope,
            lambda tx: (
                tx.run(
                    "MATCH (r:Resident {estate_id:$estate_id,resident_id:$resident_id,status:'ACTIVE'}) "
                    "RETURN count(r) AS count",
                    estate_id=scope,
                    resident_id=resident_id,
                ).single()["count"]
                == 1
            ),
        )

    def resident_units(self, who, day):
        """UI context from verified associations; uses the same eligibility gate as business actions."""
        def read(tx):
            ids = tx.run(
                "MATCH (:Resident {estate_id:$estate_id,resident_id:$resident_id,status:'ACTIVE'})"
                "-[:ASSOCIATED_WITH {estate_id:$estate_id}]->(u:Unit {estate_id:$estate_id}) "
                "RETURN DISTINCT u.unit_id AS id ORDER BY id",
                estate_id=who.estate_id, resident_id=who.actor_ref,
            ).data()
            units = []
            for item in ids:
                try:
                    row = self._eligible(tx, who.estate_id, who.actor_ref, item["id"], day, False)
                except POCError as exc:
                    if exc.code != "RESOURCE_NOT_AVAILABLE":
                        raise
                    continue
                units.append({"unit_id": row["unit"]["unit_id"], "room_no": row["unit"]["room_no"],
                              "building_name": row["building"].get("name", "模拟楼栋"),
                              "can_create_work_order": association_valid(row["association"], day, create=True)})
            return units
        return self._read(who.estate_id, read)

    def create_order(self, who, payload, payload_hash, confirmation_hash, now):
        scope = who.estate_id
        wid = f"{scope}:WO:{uuid.uuid4().hex}"
        number = f"{scope}-{now.strftime('%Y%m%d')}-{uuid.uuid4().hex[:16].upper()}"
        timestamp = now.isoformat()
        event = {
            "event_id": uuid.uuid4().hex,
            "estate_id": scope,
            "work_order_id": wid,
            "actor_ref": who.actor_ref,
            "actor_role": "RESIDENT",
            "old_status": None,
            "new_status": "SUBMITTED",
            "at": timestamp,
            "reason": "模拟住户确认报修",
            "request_id": payload["idempotency_key"],
            "expected_version": 0,
            "result_version": 1,
            "is_mock": True,
            "public_note": "报修已提交",
        }

        def create_tx(tx):
            # Resident lock serializes concurrent confirmations and idempotency checks for this actor.
            locked = tx.run(
                "MATCH (r:Resident {estate_id:$estate_id,resident_id:$resident_id,status:'ACTIVE'}) "
                "SET r._business_lock=coalesce(r._business_lock,0)+1 RETURN r.resident_id AS id",
                estate_id=scope,
                resident_id=who.actor_ref,
            ).single()
            if not locked:
                raise POCError("RESOURCE_NOT_AVAILABLE", AVAILABLE, 404)
            eligible = self._eligible(tx, scope, who.actor_ref, payload["unit_id"], now.date(), True)
            props = {
                "estate_id": scope,
                "work_order_id": wid,
                "number": number,
                "resident_id": who.actor_ref,
                "unit_id": payload["unit_id"],
                "service_id": eligible["service_id"],
                "description": payload["description"],
                "status": "SUBMITTED",
                "version": 1,
                "created_at": timestamp,
                "updated_at": timestamp,
                "idempotency_key": payload["idempotency_key"],
                "payload_hash": payload_hash,
                "confirmation_hash": confirmation_hash,
                "is_mock": True,
                "state_history_json": canonical([event]),
                "request_type": "REPAIR",
            }
            row = tx.run(
                """
                MERGE (w:WorkOrder {estate_id:$estate_id,resident_id:$resident_id,idempotency_key:$key})
                ON CREATE SET w += $props
                RETURN properties(w) AS order
                """,
                estate_id=scope,
                resident_id=who.actor_ref,
                key=payload["idempotency_key"],
                props=props,
            ).single()
            order = row["order"]
            if order["payload_hash"] != payload_hash:
                raise POCError("IDEMPOTENCY_CONFLICT", "相同幂等键不能用于不同报修内容", 409)
            if order["work_order_id"] != wid:
                return order, True
            for rel, label, key, identifier, reverse in [
                ("CREATED", "Resident", "resident_id", who.actor_ref, True),
                ("FOR_UNIT", "Unit", "unit_id", props["unit_id"], False),
                ("REQUESTS", "Service", "service_id", props["service_id"], False),
            ]:
                pattern = (
                    f"(a)-[:{rel} {{estate_id:$estate_id}}]->(w)"
                    if reverse
                    else f"(w)-[:{rel} {{estate_id:$estate_id}}]->(a)"
                )
                tx.run(
                    f"MATCH (w:WorkOrder {{estate_id:$estate_id,work_order_id:$wid}}), "
                    f"(a:{label} {{estate_id:$estate_id,{key}:$identifier}}) MERGE {pattern}",
                    estate_id=scope,
                    wid=wid,
                    identifier=identifier,
                ).consume()
            return order, False

        return self._write(scope, create_tx)

    def order_for_resident(self, who, number, day):
        def query(tx):
            rows = tx.run(
                """
                MATCH (r:Resident {estate_id:$estate_id,resident_id:$resident_id,status:'ACTIVE'})-
                      [:CREATED {estate_id:$estate_id}]->
                      (w:WorkOrder {estate_id:$estate_id,number:$number})-
                      [:FOR_UNIT {estate_id:$estate_id}]->(u:Unit {estate_id:$estate_id})
                MATCH (r)-[a:ASSOCIATED_WITH {estate_id:$estate_id}]->(u)
                MATCH (w)-[:REQUESTS {estate_id:$estate_id}]->(s:Service {estate_id:$estate_id})
                WHERE w.resident_id=r.resident_id AND w.unit_id=u.unit_id AND w.service_id=s.service_id
                  AND coalesce(w.request_type,'REPAIR')='REPAIR'
                RETURN properties(w) AS order,properties(a) AS association
                """,
                estate_id=who.estate_id,
                resident_id=who.actor_ref,
                number=number,
            ).data()
            for row in rows:
                if association_valid(row["association"], day):
                    return row["order"]
            raise POCError("WORK_ORDER_NOT_AVAILABLE", "工单不存在或无权查看", 404)

        return self._read(who.estate_id, query)

    def update_order(self, who, request, payload_hash, now):
        scope = who.estate_id

        def update_tx(tx):
            row = tx.run(
                "MATCH (w:WorkOrder {estate_id:$estate_id,number:$number}) "
                "WHERE coalesce(w.request_type,'REPAIR')='REPAIR' "
                "SET w._business_lock=coalesce(w._business_lock,0)+1 RETURN properties(w) AS order",
                estate_id=scope,
                number=request.number,
            ).single()
            if not row:
                raise POCError("WORK_ORDER_NOT_AVAILABLE", "工单不存在或无权查看", 404)
            order = row["order"]
            history = json.loads(order["state_history_json"])
            for old in history:
                if (
                    old.get("idempotency_key") == request.idempotency_key
                    and old["actor_ref"] == who.actor_ref
                ):
                    if old["payload_hash"] != payload_hash:
                        raise POCError("IDEMPOTENCY_CONFLICT", "相同幂等键不能用于不同操作", 409)
                    return order, old, True
            if order["version"] != request.expected_version:
                raise POCError("VERSION_CONFLICT", "工单已更新，请重新查看后操作", 409)
            if request.target_status not in TRANSITIONS[order["status"]]:
                raise POCError("INVALID_TRANSITION", "不允许此状态转换", 409)
            event = {
                "event_id": uuid.uuid4().hex,
                "estate_id": scope,
                "work_order_id": order["work_order_id"],
                "actor_ref": who.actor_ref,
                "actor_role": "BACKOFFICE_OPERATOR",
                "old_status": order["status"],
                "new_status": request.target_status,
                "at": now.isoformat(),
                "reason": request.reason,
                "request_id": request.idempotency_key,
                "idempotency_key": request.idempotency_key,
                "payload_hash": payload_hash,
                "expected_version": request.expected_version,
                "result_version": order["version"] + 1,
                "is_mock": True,
                "public_note": {
                    "ASSIGNED": "已安排处理",
                    "IN_PROGRESS": "正在处理",
                    "COMPLETED": "维修已完成",
                    "CANCELLED": "报修已取消",
                }[request.target_status],
            }
            history.append(event)
            updated = tx.run(
                "MATCH (w:WorkOrder {estate_id:$estate_id,number:$number}) "
                "WHERE w.version=$expected SET w.status=$status,w.version=$version,w.updated_at=$at,"
                "w.state_history_json=$history RETURN properties(w) AS order",
                estate_id=scope,
                number=request.number,
                expected=request.expected_version,
                status=request.target_status,
                version=event["result_version"],
                at=event["at"],
                history=canonical(history),
            ).single()
            if not updated:
                raise POCError("VERSION_CONFLICT", "工单已更新，请重新查看后操作", 409)
            return updated["order"], event, False

        return self._write(scope, update_tx)

    def semantic_index_rows(self, scope):
        def query(tx):
            result = tx.run("""
                MATCH (p:Policy {estate_id:$estate_id})
                RETURN p.policy_id AS id,p.estate_id AS estate_id,p.question AS question,p.title AS title,
                       p.service_code AS service_code,p.version_hash AS version_hash,p._record_json AS record
                UNION ALL
                MATCH (f:FAQ {estate_id:$estate_id})
                RETURN f.faq_id AS id,f.estate_id AS estate_id,f.question AS question,f.question AS title,
                       f.service_code AS service_code,f.version_hash AS version_hash,f._record_json AS record
                """, estate_id=scope).data()
            rows = []
            for r in result:
                if not isinstance(r["id"], str) or not r["id"].startswith(scope + ":"):
                    continue
                try:
                    original = json.loads(r.pop("record") or "{}")
                except (ValueError, TypeError):
                    original = {}
                fact = original.get("fact", {})
                r["concepts"] = " ".join(str(fact.get(k) or "") for k in ("subject", "predicate", "fact_key"))
                rows.append(r)
            return rows
        return self._read(scope, query)

    def knowledge_bundles(self, scope, service_code=None):
        def query(tx):
            approved = tx.run(
                "MATCH (p:Policy {estate_id:$estate_id,status:'APPROVED',review_status:'APPROVED'}) RETURN count(p) AS count",
                estate_id=scope,
            ).single()["count"]
            if not approved:
                return []
            return tx.run(
                """
                MATCH (p:Policy {estate_id:$estate_id,status:'APPROVED',review_status:'APPROVED'})-
                      [:APPLIES_TO {estate_id:$estate_id}]->(e:Estate {estate_id:$estate_id})
                MATCH (p)-[:GOVERNS {estate_id:$estate_id}]->(s:Service {estate_id:$estate_id,enabled:true})
                MATCH (p)-[:SUPPORTED_BY {estate_id:$estate_id}]->
                      (c:EvidenceChunk {estate_id:$estate_id})<-
                      [:HAS_CHUNK {estate_id:$estate_id}]-(d:Document {estate_id:$estate_id})
                WHERE ($service_code IS NULL OR s.service_code=$service_code)
                  AND p.service_id=s.service_id AND c.document_id=d.document_id
                OPTIONAL MATCH (rd:ReviewDecision {estate_id:$estate_id})-
                      [:REVIEWS {estate_id:$estate_id}]->(p)
                OPTIONAL MATCH (dr:ReviewDecision {estate_id:$estate_id})-[:REVIEWS {estate_id:$estate_id}]->(d)
                  WHERE dr.review_decision_id=d.demo_review_decision_id
                OPTIONAL MATCH (cr:ReviewDecision {estate_id:$estate_id})-[:REVIEWS {estate_id:$estate_id}]->(c)
                  WHERE cr.review_decision_id=c.demo_review_decision_id
                RETURN properties(p) AS policy, collect(DISTINCT {chunk:properties(c),document:properties(d),
                       document_review:properties(dr),chunk_review:properties(cr)}) AS evidence,
                       collect(DISTINCT properties(rd)) AS reviews
                """,
                estate_id=scope,
                service_code=service_code,
            ).data()

        return self._read(scope, query)

    def faq_bundles(self, scope, service_code=None, *, knowledge_mode="current"):
        if knowledge_mode not in {"current", "demo"}:
            raise ValueError("FAQ查询用途无效")
        return self._read(scope, lambda tx: tx.run(
            """
            MATCH (f:FAQ {estate_id:$estate_id})-
                  [:APPLIES_TO {estate_id:$estate_id}]->(:Estate {estate_id:$estate_id})
            MATCH (f)-[:GOVERNS {estate_id:$estate_id}]->(s:Service {estate_id:$estate_id,enabled:true})
            MATCH (f)-[:SUPPORTED_BY {estate_id:$estate_id}]->(c:EvidenceChunk {estate_id:$estate_id})<-
                  [:HAS_CHUNK {estate_id:$estate_id}]-(d:Document {estate_id:$estate_id})
            WHERE f.service_id=s.service_id AND f.chunk_id=c.chunk_id AND f.document_id=d.document_id
                  AND c.document_id=d.document_id AND ($service_code IS NULL OR s.service_code=$service_code)
                  AND (($mode='current' AND f.review_status='APPROVED') OR
                       ($mode='demo' AND f.demo_review_status='APPROVED'))
            OPTIONAL MATCH (r:ReviewDecision {estate_id:$estate_id})-[:REVIEWS {estate_id:$estate_id}]->(f)
            OPTIONAL MATCH (cr:ReviewDecision {estate_id:$estate_id})-[:REVIEWS {estate_id:$estate_id}]->(c)
            OPTIONAL MATCH (dr:ReviewDecision {estate_id:$estate_id})-[:REVIEWS {estate_id:$estate_id}]->(d)
            OPTIONAL MATCH (f)-[ar:ANSWERED_BY {estate_id:$estate_id}]->(ap:Policy {estate_id:$estate_id})
            RETURN properties(f) AS faq,properties(c) AS chunk,properties(d) AS document,
                   collect(DISTINCT properties(r)) AS reviews,
                   collect(DISTINCT properties(cr)) AS chunk_reviews,collect(DISTINCT properties(dr)) AS document_reviews,
                   collect(DISTINCT {policy_id:ap.policy_id,version_hash:ap.version_hash,selection_hash:ar.selection_hash}) AS selection_policies
            """, estate_id=scope, service_code=service_code, mode=knowledge_mode,
        ).data())

    def pending_faq_questions(self, scope, service_code=None):
        """Read pending question titles only; never load candidate answers into resident replies."""
        return self._read(scope, lambda tx: tx.run(
            """
            MATCH (f:FAQ {estate_id:$estate_id,review_status:'PENDING_REVIEW'})-
                  [:APPLIES_TO {estate_id:$estate_id}]->(:Estate {estate_id:$estate_id})
            MATCH (f)-[:GOVERNS {estate_id:$estate_id}]->(s:Service {estate_id:$estate_id,enabled:true})
            WHERE f.service_id=s.service_id AND ($service_code IS NULL OR s.service_code=$service_code)
            RETURN f.estate_id AS estate_id,f.faq_id AS faq_id,f.question AS question,f.service_code AS service_code,
                   f.demo_review_status AS demo_review_status
            ORDER BY f.faq_id
            """, estate_id=scope, service_code=service_code,
        ).data())

    def stats(self, scope):
        estate(scope)

        def query(tx):
            result = {}
            for label in CORE_IDS:
                result[label] = tx.run(
                    f"MATCH (n:{label} {{estate_id:$estate_id}}) RETURN count(n) AS count", estate_id=scope
                ).single()["count"]
            result["pending_policies"] = tx.run(
                "MATCH (p:Policy {estate_id:$estate_id,review_status:'PENDING_REVIEW'}) "
                "RETURN count(p) AS count",
                estate_id=scope,
            ).single()["count"]
            result["demo_approved_policies"] = tx.run(
                "MATCH (p:Policy {estate_id:$estate_id,status:'APPROVED',review_status:'APPROVED',review_purpose:'POC_DEMO'}) "
                "RETURN count(p) AS count",
                estate_id=scope,
            ).single()["count"]
            result["pending_faqs"] = tx.run(
                "MATCH (f:FAQ {estate_id:$estate_id,review_status:'PENDING_REVIEW'}) RETURN count(f) AS count",
                estate_id=scope,
            ).single()["count"]
            result["demo_approved_faqs"] = tx.run(
                "MATCH (f:FAQ {estate_id:$estate_id,demo_review_status:'APPROVED'}) RETURN count(f) AS count",
                estate_id=scope,
            ).single()["count"]
            for name, source in (("day1_selected_faqs", "DAY1_POLICY"), ("supplement_selected_faqs", "SUPPLEMENT_QA")):
                result[name] = tx.run("MATCH (f:FAQ {estate_id:$estate_id,answer_selection_source:$source}) RETURN count(f) AS count",
                                      estate_id=scope, source=source).single()["count"]
            return result

        return self._read(scope, query)
