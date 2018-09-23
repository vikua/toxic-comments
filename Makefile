.PHONY: keras-install keras-train keras-predict keras-test


FLAGS=


keras-install: 
	pip install -r keras/requirements.txt
	@echo
	@echo "keras toxic comments model dependencies installed"
	@echo


keras-train: 
	python ./keras/run.py train --input-path ./data --output-path ./bin --epochs 20 \
		--batch-size 128 --max-features 60000


keras-predict: 
	python ./keras/run.py predict --model-path ./bin/keras.pkl 


keras-test:
	pytest -s -v $(FLAGS) ./keras/tests/
