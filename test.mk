include Makefile

CMD_PREPARE=\
  export DEBIAN_FRONTEND=noninteractive && \
  apt-get -qq update && \
  apt-get -qq install -y --no-install-recommends pandoc >/dev/null && \
  $(PREPARE_DATA_CMD)

CMD_NBCONVERT=\
  jupyter nbconvert \
  --execute \
  --no-prompt \
  --no-input \
  --to=asciidoc \
  --ExecutePreprocessor.timeout=600 \
  --output=/tmp/out \
  $(PROJECT_PATH_ENV)/$(NOTEBOOKS_DIR)/demo.ipynb && \
  echo "Test succeeded: PROJECT_PATH_ENV=$(PROJECT_PATH_ENV) TRAINING_MACHINE_TYPE=$(TRAINING_MACHINE_TYPE)"

SUCCESS_MSG=Test succeeded: PROJECT_PATH_ENV=$(PROJECT_PATH_ENV) TRAINING_MACHINE_TYPE=$(TRAINING_MACHINE_TYPE)


.PHONY: test_jupyter
test_jupyter: JUPYTER_CMD=bash -c '$(CMD_PREPARE) && $(CMD_NBCONVERT)'
test_jupyter: jupyter
	@# This is a workaround for https://github.com/neuromation/platform-client-python/issues/1470
	$(NEURO) status $(JUPYTER_JOB) | tee /dev/stderr | grep -q "Exit code: 0"
	@echo $(SUCCESS_MSG)
	$(MAKE) kill-jupyter

.PHONY: test_jupyter_baked
test_jupyter_baked: PROJECT_PATH_ENV=/project-local
test_jupyter_baked:
	$(NEURO) run $(RUN_EXTRA) \
		--name $(JOB_NAME) \
		--preset $(TRAINING_MACHINE_TYPE) \
		$(CUSTOM_ENV_NAME) \
		bash -c '$(CMD_PREPARE) && $(CMD_NBCONVERT)'
	$(NEURO) status $(JUPYTER_JOB) | tee /dev/stderr | grep -q "Exit code: 0"
	@echo $(SUCCESS_MSG)
	$(NEURO) kill $(JOB_NAME)