THE_OS := $(shell uname -s)

default:
	@echo "Detected OS: ${THE_OS}"
	$(MAKE) CC=cc CXX=CC \
		CXXFLAGS='$(CXXFLAGS) -Wall -Wextra -pipe -O3 -g -ffast-math -flto -march=native -std=c++14 -DNDEBUG'  \
		LDFLAGS='$(LDFLAGS) -flto -g' \
		leelaz

debug:
	@echo "Detected OS: ${THE_OS}"
	$(MAKE) CC=gcc CXX=g++ \
		CXXFLAGS='$(CXXFLAGS) -Wall -Wextra -pipe -Og -g -std=c++14' \
		LDFLAGS='$(LDFLAGS) -g' \
		leelaz

clang:
	@echo "Detected OS: ${THE_OS}"
	$(MAKE) CC=clang-5.0 CXX=clang++-5.0 \
		CXXFLAGS='$(CXXFLAGS) -Wall -Wextra -Wno-missing-braces -O3 -ffast-math -flto -march=native -std=c++14 -DNDEBUG' \
		LDFLAGS='$(LDFLAGS) -flto -fuse-linker-plugin' \
		leelaz

DYNAMIC_LIBS = -lboost_program_options -lpthread -lz -lcuda -lcudart -lcudnn
LIBS =

ifeq ($(THE_OS),Linux)
# for Linux with OpenBLAS
	CUDNN = /usr/local/cuda/include
	CUDART = /usr/local/cuda/lib64
	CXXFLAGS += -I/usr/include/openblas -I$(CUDNN)
	DYNAMIC_LIBS += -lopenblas
	#DYNAMIC_LIBS += -lOpenCL
	DYNAMIC_LIBS += -L$(CUDART)
endif
ifeq ($(THE_OS),Darwin)
# for macOS (comment out the Linux part)
	LIBS += -framework Accelerate
	LIBS += -framework OpenCL
	CXXFLAGS += -I/System/Library/Frameworks/Accelerate.framework/Versions/Current/Headers
endif

# for MKL instead of OpenBLAS
#DYNAMIC_LIBS += -lmkl_rt
#CXXFLAGS += -I/opt/intel/mkl/include
#LDFLAGS  += -L/opt/intel/mkl/lib/intel64/

CXXFLAGS += -I.
CPPFLAGS += -MD -MP
CXXFLAGS += -I/N/u/xu56/BigRed2/openblas/include -I/N/u/xu56/BigRed2
LDFLAGS += -L/N/u/xu56/BigRed2/openblas/lib -L/N/soft/cle5/boost/1.65.0/lib

sources = Network.cpp FullBoard.cpp KoState.cpp Training.cpp \
	  TimeControl.cpp UCTSearch.cpp GameState.cpp Leela.cpp \
	  SGFParser.cpp Timing.cpp Utils.cpp FastBoard.cpp \
	  SGFTree.cpp Zobrist.cpp FastState.cpp GTP.cpp Random.cpp \
	  SMP.cpp UCTNode.cpp OpenCL.cpp OpenCLScheduler.cpp \
	  NNCache.cpp Tuner.cpp CuDNN.cpp

objects = $(sources:.cpp=.o)
deps = $(sources:%.cpp=%.d)

-include $(deps)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c -o $@ $<

leelaz: $(objects)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LIBS) $(DYNAMIC_LIBS)

clean:
	-$(RM) leelaz $(objects) $(deps)

.PHONY: clean default debug clang
