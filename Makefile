.PHONY: test audit release-check synthetic

test:
	PYTHONPATH=src python -m pytest -q

audit:
	python tools/audit_legacy_code.py --root .. \
		--csv docs/legacy_code_inventory.csv \
		--markdown docs/LEGACY_CODE_AUDIT.md \
		--exclude NBT_submission_DRAFT_2026-09-09 \
		--exclude NanoHELDI_MS_GitHub_ready_2026-09-09 \
		--exclude NanoHELDI_MS_INTERNAL_AUDIT_2026-09-09 \
		--redact-paths

release-check:
	python tools/release_check.py

synthetic:
	python examples/make_synthetic_data.py --output examples/data --overwrite
