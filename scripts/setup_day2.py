"""Explicit Day 2 initialization of the selected project instance; no approval and no reset."""

import argparse
import hashlib
import json

from property_ontology.actions import PropertyActions
from property_ontology.faq_importer import FAQ_PACKET, import_faq_candidates
from property_ontology.importer import import_knowledge, seed_mock
from property_ontology.runtime import RuntimeSettings


def main(profile):
    settings = RuntimeSettings.load(profile=profile)
    app = PropertyActions(settings)
    try:
        app._repo.initialize_schema()
        imported = import_knowledge(app._repo, settings.root)
        mocked = seed_mock(app._repo, app.auth, settings.root)
        faq_imported = (import_faq_candidates(app._repo, settings.root)
                        if (settings.root / FAQ_PACKET / "candidates.json").exists() else None)
        result = {
            "profile": profile,
            "import": imported,
            "faq_import": faq_imported,
            "mock": mocked["mock_objects"],
            "counts": {e: app._repo.stats(e) for e in ["E001", "E002"]},
        }
        out = settings.root / "data/day2"
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{profile}_import_report.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n"
        )
        # Protect the entire prior alignment packet and the original evidence/review artifacts.
        baseline = out / "protected_inputs.json"
        if not baseline.exists():
            paths = [p for p in (settings.root / "data/raw").rglob("*") if p.is_file()]
            paths += [
                p for p in (settings.root / "outputs/schema_alignment_20261027").rglob("*") if p.is_file()
            ]
            paths += [settings.root / "data/day1/review_queue.json"]
            baseline.write_text(
                json.dumps(
                    {
                        str(p.relative_to(settings.root)): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in paths
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n"
            )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        app.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--profile", choices=["main", "test"], default="main")
    main(p.parse_args().profile)
