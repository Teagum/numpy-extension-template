mod_name.so:
	python3 setup.py build_ext --inplace

test:
	python3 test.py
