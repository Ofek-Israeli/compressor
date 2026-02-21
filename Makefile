# Phase 0 config and data collection
# See docs/phase0.md

CONFIG ?= config.yaml
PYTHON ?= python3

.PHONY: menuconfig collect help

# Interactive menu to create or edit config.yaml (or CONFIG=...)
menuconfig:
	$(PYTHON) scripts/menuconfig.py $(CONFIG)

# Run Phase 0 data collection (requires valid config)
collect:
	$(PYTHON) collect_data_financebench.py --config $(CONFIG)

collect-resume:
	$(PYTHON) collect_data_financebench.py --config $(CONFIG) --resume

help:
	@echo "Targets:"
	@echo "  make menuconfig      - Create/edit config (default: config.yaml)"
	@echo "  make collect         - Run Phase 0 data collection"
	@echo "  make collect-resume - Resume from checkpoints"
	@echo "  make CONFIG=path.yaml menuconfig  - Edit specific config file"
