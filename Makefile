# Phase 0 config and data collection
# See docs/phase0.md. Kconfig works like learning_grammar.

CONFIG_DIR := config
KCONFIG := $(CONFIG_DIR)/Kconfig
CONFIG ?= $(CONFIG_DIR)/.config
DEFCONFIG := $(CONFIG_DIR)/presets/defconfig
PYTHON ?= python3

.PHONY: menuconfig menuconfig-tui defconfig savedefconfig collect collect-resume help

# Interactive configuration (Kconfig TUI, like learning_grammar)
menuconfig:
	@$(PYTHON) -c "from kconfiglib import Kconfig; import menuconfig; \
		import os; os.environ['KCONFIG_CONFIG'] = '$(CONFIG)'; \
		menuconfig.menuconfig(Kconfig('$(KCONFIG)'))"

# Alias for menuconfig (same target)
menuconfig-tui: menuconfig

# Load default configuration
defconfig:
	@if [ -f $(DEFCONFIG) ]; then \
		cp $(DEFCONFIG) $(CONFIG); \
		echo "Loaded default configuration from $(DEFCONFIG)"; \
	else \
		echo "Error: $(DEFCONFIG) not found"; \
		exit 1; \
	fi

# Save current config as defconfig
savedefconfig:
	@if [ -f $(CONFIG) ]; then \
		cp $(CONFIG) $(DEFCONFIG); \
		echo "Saved current configuration to $(DEFCONFIG)"; \
	else \
		echo "Error: $(CONFIG) not found"; \
		exit 1; \
	fi

# Run Phase 0 data collection (requires valid config)
collect:
	$(PYTHON) collect_data_financebench.py --config $(CONFIG)

collect-resume:
	$(PYTHON) collect_data_financebench.py --config $(CONFIG) --resume

help:
	@echo "Targets:"
	@echo "  make menuconfig     - Interactive Kconfig menu (edit $(CONFIG))"
	@echo "  make defconfig      - Load default configuration from $(DEFCONFIG)"
	@echo "  make savedefconfig  - Save current config as defconfig"
	@echo "  make collect        - Run Phase 0 data collection"
	@echo "  make collect-resume - Resume from checkpoints"
	@echo "  make CONFIG=path collect - Use specific .config file"
