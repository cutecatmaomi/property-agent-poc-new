"""Archived command name: budget helper retained, old question-bank run retired."""
import json
import time
from property_ontology.domain import POCError

class Budget:
    def __init__(self, path, maximum):
        self.path, self.maximum = path, maximum
        self.attempts = json.loads(path.read_text()) if path.exists() else []

    def call(self, client, payload):
        if len(self.attempts) >= self.maximum:
            raise POCError('EVAL_BUDGET_EXHAUSTED', '本轮测速达到调用上限。', 503)
        row = {'number': len(self.attempts) + 1,
               'function': payload['tools'][0]['function']['name'], 'status': 'STARTED',
               'input_chars': len(json.dumps(payload, ensure_ascii=False))}
        self.attempts.append(row)
        # Reserve *before* the request, even if the process disconnects/crashes.
        self.save()
        started = time.perf_counter()
        try:
            result = client._http(payload)
            row.update(status='SUCCEEDED', usage=result.get('usage', {}))
            return result
        except Exception as exc:
            row.update(status='FAILED', error_code=getattr(exc, 'code', type(exc).__name__))
            raise
        finally:
            row['seconds'] = round(time.perf_counter() - started, 4)
            self.save()

    def save(self):
        self.path.write_text(json.dumps(self.attempts, ensure_ascii=False, indent=2))


def main():
    raise SystemExit('旧题库测速已停用。当前题库为 GoldenQuestions.xlsx；如需付费自然问法复测，请先授权预算，再用 scripts/evaluate_package_golden.py --help。')

if __name__ == '__main__':
    main()
