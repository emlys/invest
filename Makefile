
ENV = "./env"

SHELL := /bin/bash

# Output directory names
DIST_DIR := dist
DIST_DATA_DIR := $(DIST_DIR)/data
BUILD_DIR := build


# Target names.
EXAMPLE_BINARIES_DIR := $(DIST_DIR)/example

MAC_DISK_IMAGE_FILE := "$(DIST_DIR)/Example.dmg"
MAC_BINARIES_ZIP_FILE := "$(DIST_DIR)/Example-mac.zip"
MAC_APPLICATION_BUNDLE := "$(BUILD_DIR)/mac_app/InVEST.app"


$(BUILD_DIR) $(DATA_DIR) $(DIST_DIR) $(DIST_DATA_DIR):
	mkdir -p $@


# Python conda environment management
env:
	$(info make env...)
	@echo "NOTE: requires 'requests' be installed in base Python"
	python3 ./scripts/convert-requirements-to-conda-yml.py requirements.txt requirements-dev.txt requirements-gui.txt > requirements-all.yml
	conda create -p $(ENV) -y -c conda-forge python=3.8 nomkl
	conda env update -p $(ENV) --file requirements-all.yml
	@echo "----------------------------"
	@echo "To activate the new conda environment and install example:"
	@echo ">> conda activate $(ENV)"
	@echo ">> make install"


# compatible with pip>=7.0.0
# REQUIRED: Need to remove example.egg-info directory so recent versions
# of pip don't think CWD is a valid package.
install: $(DIST_DIR)/example%.whl
	$(info make install...)
	rm -r example.egg-info
	python3 -m pip install --isolated --upgrade --no-index --only-binary example --find-links=dist example


# Bulid python packages and put them in dist/
python_packages: $(DIST_DIR)/example%.whl
$(DIST_DIR)/example%.whl: | $(DIST_DIR)
	$(info make python_packages...)
	python3 setup.py bdist_wheel

# Build binaries and put them in dist/example
binaries: $(EXAMPLE_BINARIES_DIR)
$(EXAMPLE_BINARIES_DIR): | $(DIST_DIR) $(BUILD_DIR)
	$(info make binaries...)
	python3 -m PyInstaller --workpath $(BUILD_DIR)/pyi-build --clean --distpath $(DIST_DIR) example.spec
	conda list --export > $(EXAMPLE_BINARIES_DIR)/package_versions.txt

DMG_CONFIG_FILE := dmgconf.py
mac_dmg: $(MAC_DISK_IMAGE_FILE)
$(MAC_DISK_IMAGE_FILE): $(DIST_DIR) $(MAC_APPLICATION_BUNDLE) $(USERGUIDE_TARGET_DIR)
	$(info make mac_dmg...)
	dmgbuild -Dexampledir=$(MAC_APPLICATION_BUNDLE) -s $(DMG_CONFIG_FILE) "Example" $(MAC_DISK_IMAGE_FILE)

USERGUIDE_TARGET_DIR := $(DIST_DIR)/userguide
mac_app: $(MAC_APPLICATION_BUNDLE)
$(MAC_APPLICATION_BUNDLE): $(BUILD_DIR) $(EXAMPLE_BINARIES_DIR)
	$(info make mac_app...)
	@echo "MAC_APPLICATION_BUNDLE:"
	@echo $(MAC_APPLICATION_BUNDLE)
	./build_app_bundle.sh $(EXAMPLE_BINARIES_DIR) $(USERGUIDE_TARGET_DIR) $(MAC_APPLICATION_BUNDLE)

