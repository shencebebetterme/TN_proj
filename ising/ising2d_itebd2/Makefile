LIBRARY_DIR=/data/scbebetterme/GitRepo/ITensor

APP=ising2d_itebd2

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
	$(CCCOM) -c $(CCFLAGS) -o $@ $< -I /data/scbebetterme/GitRepo/C++/armadillo-code/include -DARMA_DONT_USE_WRAPPER

.debug_objs/%.o: %.cc $(HEADERS) $(TENSOR_HEADERS)
	$(CCCOM) -c $(CCGFLAGS) -o $@ $< -I /data/scbebetterme/GitRepo/C++/armadillo-code/include -DARMA_DONT_USE_WRAPPER

#Targets -----------------

build: $(APP)
debug: $(APP)-g

$(APP): $(OBJECTS) $(ITENSOR_LIBS)
	$(CCCOM) $(CCFLAGS) $(OBJECTS) -o $(APP).out $(LIBFLAGS) -I /data/scbebetterme/GitRepo/C++/armadillo-code/include -DARMA_DONT_USE_WRAPPER

$(APP)-g: mkdebugdir $(GOBJECTS) $(ITENSOR_GLIBS)
	$(CCCOM) $(CCGFLAGS) $(GOBJECTS) -o $(APP)-g.out $(LIBGFLAGS) -I /data/scbebetterme/GitRepo/C++/armadillo-code/include -DARMA_DONT_USE_WRAPPER

clean:
	rm -fr .debug_objs *.o $(APP) $(APP)-g $(APP).out $(APP)-g.out

mkdebugdir:
	mkdir -p .debug_objs

