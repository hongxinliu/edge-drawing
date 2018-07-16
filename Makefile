all:
	mkdir -p build && cd build && cmake .. -DCMAKE_INSTALL_PREFIX=. && make && make install
