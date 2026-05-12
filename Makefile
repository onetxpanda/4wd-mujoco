.PHONY: all raw meshes model test clean

ZIP_URL := https://files.waveshare.com/upload/2/23/DDSM115_4WD-A.zip
ZIP := raw/DDSM115_4WD-A.zip
STEP_STAMP := raw/.extracted
STEP_FILE := raw/DDSM115 4WD-A.step
MESH_STAMP := meshes/assembly.json

all: model

raw: $(STEP_STAMP)

$(ZIP):
	mkdir -p raw
	curl -L -o $@ $(ZIP_URL)

$(STEP_STAMP): $(ZIP)
	unzip -o $(ZIP) -d raw/
	touch $@

meshes: $(MESH_STAMP)

$(MESH_STAMP): $(STEP_STAMP) step_to_meshes.py
	python3 step_to_meshes.py "$(STEP_FILE)" meshes

model: ddsm115_4wd.xml

ddsm115_4wd.xml: $(MESH_STAMP) build_model.py
	python3 build_model.py meshes ddsm115_4wd.xml assets

test: ddsm115_4wd.xml test_model.py
	python3 test_model.py

clean:
	rm -rf meshes assets ddsm115_4wd.xml preview.png
