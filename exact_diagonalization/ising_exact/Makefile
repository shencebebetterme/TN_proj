
### uncomment the following two lines on server
#LIBRARY_DIR=/data/sc/GitRepo/ITensor
#ARMA_FLAGS=-I/data/sc/GitRepo/C++/armadillo-code/include -DARMA_DONT_USE_WRAPPER -larpack

### uncomment the following two lines on wsl
LIBRARY_DIR=/home/ce18/ITensor
ARMA_FLAGS=-I/home/ce18/armadillo-code/include -DARMA_DONT_USE_WRAPPER -larpack


APP=fibo_TRG_exact

CCFILES=$(APP).cc

#################################################################
#################################################################
#################################################################
#################################################################


include $(LIBRARY_DIR)/this_dir.mk
include $(LIBRARY_DIR)/options.mk

TENSOR_HEADERS=$(LIBRARY_DIR)/itensor/all.h

#Mappings --------------
OBJECTS=$(patsubst %.cc,%.o, $(CCFILES))
GOBJECTS=$(patsubst %,.debug_objs/%, $(OBJECTS))

#Rules ------------------

%.o: %.cc $(HEADERS) $(TENSOR_HEADERS)
	$(CCCOM) -c $(CCFLAGS) -o $@ $< $(ARMA_FLAGS) 
.debug_objs/%.o: %.cc $(HEADERS) $(TENSOR_HEADERS)
	$(CCCOM) -c $(CCGFLAGS) -o $@ $<

#Targets -----------------

build: $(APP)
debug: $(APP)-g

$(APP): $(OBJECTS) $(ITENSOR_LIBS)
	$(CCCOM) $(CCFLAGS) $(OBJECTS) -o $(APP).out $(LIBFLAGS) $(ARMA_FLAGS)

$(APP)-g: mkdebugdir $(GOBJECTS) $(ITENSOR_GLIBS)
	$(CCCOM) $(CCGFLAGS) $(GOBJECTS) -o $(APP)-g $(LIBGFLAGS)

clean:
	rm -fr .debug_objs *.o $(APP).out $(APP)-g

mkdebugdir:
	mkdir -p .debug_objs

