
ENV = "./env"

SHELL := /bin/bash

VERSION := $(shell python3 setup.py --version)

# Output directory names
DIST_DIR := dist
DIST_DATA_DIR := $(DIST_DIR)/data
BUILD_DIR := build


# Target names.
INVEST_BINARIES_DIR := $(DIST_DIR)/invest

MAC_DISK_IMAGE_FILE := "$(DIST_DIR)/InVEST_$(VERSION).dmg"
MAC_BINARIES_ZIP_FILE := "$(DIST_DIR)/InVEST-$(VERSION)-mac.zip"
MAC_APPLICATION_BUNDLE := "$(BUILD_DIR)/mac_app_$(VERSION)/InVEST.app"


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
	@echo "To activate the new conda environment and install natcap.invest:"
	@echo ">> conda activate $(ENV)"
	@echo ">> make install"


# compatible with pip>=7.0.0
# REQUIRED: Need to remove natcap.invest.egg-info directory so recent versions
# of pip don't think CWD is a valid package.
install: $(DIST_DIR)/natcap.invest%.whl
	$(info make install...)
	rm -r natcap.invest.egg-info
	python3 -m pip install --isolated --upgrade --no-index --only-binary natcap.invest --find-links=dist natcap.invest


# Bulid python packages and put them in dist/
python_packages: $(DIST_DIR)/natcap.invest%.whl $(DIST_DIR)/natcap.invest%.zip
$(DIST_DIR)/natcap.invest%.whl: | $(DIST_DIR)
	$(info make python_packages...)
	python3 setup.py bdist_wheel

$(DIST_DIR)/natcap.invest%.zip: | $(DIST_DIR)
	python3 setup.py sdist --formats=zip


# Build binaries and put them in dist/invest
# The `invest list` is to test the binaries.  If something doesn't
# import, we want to know right away.  No need to provide the `.exe` extension
# on Windows as the .exe extension is assumed.
binaries: $(INVEST_BINARIES_DIR)
$(INVEST_BINARIES_DIR): | $(DIST_DIR) $(BUILD_DIR)
	$(info make binaries...)
	python3 -m PyInstaller --workpath $(BUILD_DIR)/pyi-build --clean --distpath $(DIST_DIR) exe/invest.spec
	conda list --export > $(INVEST_BINARIES_DIR)/package_versions.txt

DMG_CONFIG_FILE := installer/darwin/dmgconf.py
mac_dmg: $(MAC_DISK_IMAGE_FILE)
$(MAC_DISK_IMAGE_FILE): $(DIST_DIR) $(MAC_APPLICATION_BUNDLE) $(USERGUIDE_TARGET_DIR)
	$(info make mac_dmg...)
	dmgbuild -Dinvestdir=$(MAC_APPLICATION_BUNDLE) -s $(DMG_CONFIG_FILE) "InVEST $(VERSION)" $(MAC_DISK_IMAGE_FILE)

mac_app: $(MAC_APPLICATION_BUNDLE)
$(MAC_APPLICATION_BUNDLE): $(BUILD_DIR) $(INVEST_BINARIES_DIR) $(USERGUIDE_TARGET_DIR)
	$(info make mac_app...)
	./installer/darwin/build_app_bundle.sh $(VERSION) $(INVEST_BINARIES_DIR) $(USERGUIDE_TARGET_DIR) $(MAC_APPLICATION_BUNDLE)

mac_zipfile: $(MAC_BINARIES_ZIP_FILE)
$(MAC_BINARIES_ZIP_FILE): $(DIST_DIR) $(MAC_APPLICATION_BUNDLE) $(USERGUIDE_TARGET_DIR)
	$(info make mac_zipfile...)
	./installer/darwin/build_zip.sh $(VERSION) $(MAC_APPLICATION_BUNDLE) $(USERGUIDE_TARGET_DIR)

