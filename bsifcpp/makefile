CXX?=g++

CXXFLAGS=-O0 -std=c++11 -Wall -fmessage-length=0 \
					-g -gdwarf-3 -Wformat  -fPIC
INCLUDES=-I/home/allansp/local/opencv-3.2.0/include
LINKFLAGS=-L/home/allansp/local/opencv-3.2.0/lib \
					-lopencv_imgproc \
					-lopencv_core \
					-lopencv_highgui \
					-lopencv_imgcodecs

# this is to link the wrapper module
MODULEINCLUDES:=$(INCLUDES) -I/home/allansp/local/miniconda3/envs/dl-pa-env/include/boost/ \
					-I/home/allansp/local/miniconda3/envs/dl-pa-env/include/python3.5m/
MODULELINKFLAGS:=$(LINKFLAGS) -L/home/allansp/local/miniconda3/envs/dl-pa-env/lib \
					-lpython3.5m \
					-L/home/allansp/local/miniconda3/envs/dl-pa-env/lib \
					-lboost_python \
					-lboost_numpy3


all: libbsif.so

libbsif.so: filtermap.o bsif.o bsifmodule.o
	$(CXX) -shared $(MODULEINCLUDES) -o $@ $(CXXFLAGS) $^ $(MODULELINKFLAGS)

bsif.o: filtermap.o bsif.cpp
	$(CXX) -c $(INCLUDES) -o $@ $(CXXFLAGS) $^ $(LINKFLAGS)

test: test.cpp
	$(CXX) $(MODULEINCLUDES) -o $@ $(CXXFLAGS) $^ -L. -lbsif $(MODULELINKFLAGS)

bsifmodule.o: bsif.o bsifmodule.cpp
	$(CXX) -c $(MODULEINCLUDES) -o $@ $(CXXFLAGS) $^ $(MODULELINKFLAGS)

filtermap.o: filtermap.cpp
	$(CXX) -c $(INCLUDES) -o $@ $(CXXFLAGS) $^ $(LINKFLAGS)

clean:
	rm -f *o bsifcpp test
