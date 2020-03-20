CXX = g++
#HEADERS     = potential.h common.h
CXXFLAGS    = -O3 -fPIC -fopenmp 

all: libobj.so

objfunction.o: objfunction.cc #$(HEADERS)

libobj.so: $(HEADERS) objfunction.o
	$(CXX) $(CXXFLAGS) -shared -Wl,-install_name,libobj.so -o libobj.so objfunction.o

clean:
	@rm -f *.o libobj.so

