# Phase 0 config and data collection
# See docs/phase0.md

CONFIG ?= config.yaml
PYTHON ?= python3

.PHONY: menuconfig collect help

# GUI to create or edit config.yaml (or CONFIG=...)
menuconfig:
	$(PYTHON) scripts/menuconfig.py $(CONFIG)

# Terminal UI (no GUI)
menuconfig-tui:
	$(PYTHON) scripts/menuconfig.py $(CONFIG) --tui

# Run Phase 0 data collection (requires valid config)
collect:
	$(PYTHON) collect_data_financebench.py --config $(CONFIG)

collect-resume:
	$(PYTHON) collect_data_financebench.py --config $(CONFIG) --resume

help:
	@echo "Targets:"
	@echo "  make menuconfig      - Open GUI to create/edit config (default: config.yaml)"
	@echo "  make menuconfig-tui  - Terminal menu instead of GUI"
	@echo "  make collect        - Run Phase 0 data collection"
	@echo "  make collect-resume - Resume from checkpoints"
	@echo "  make CONFIG=path.yaml menuconfig  - Edit specific config file"
