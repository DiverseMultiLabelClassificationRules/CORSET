build: setup.py corset/samplers/dfs.pyx
	CC=gcc CXX=g++ python setup.py build_ext --inplace
ann:
	cython -a corset/samplers/dfs.pyx
clean:
	rm -rf build
	rm corset/samplers/*.so


test:
	pytest corset/tests

prof: corset/greedy/divmax_v2.py
	kernprof -l greedy_demo.py
	python -m line_profiler greedy_demo.py.lprof    


sync:
	git add mlruns
	git commit -m "mlflow results sync"
	git push
