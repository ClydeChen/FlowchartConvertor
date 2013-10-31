CXX = g++
CFLAGS = `pkg-config --cflags opencv`
LIBS = `pkg-config --libs opencv`

#% : %.cpp
#	CXX $(CFLAGS) $(LIBS) -o $@ $<

FC_SOURCES = main.cpp FlowchartConvertor.cpp
FC_OBJECTS = main.o FlowchartConvertor.o


FLOWCHART_CONVERTOREXE = flowchartconvertor


all: clean flowchartconvertor

clean:
	rm -rf $(FLOWCHART_CONVERTOREXE) $(FC_OBJECTS)
	@echo 'Clean finished.'

flowchartconvertor: $(FC_OBJECTS)
	@echo 'Building FLOWCHART_CONVERTOREXE'
	$(CXX) -o $(FLOWCHART_CONVERTOREXE) $(FC_OBJECTS) $(LIBS)
	@echo 'Build finished.'

%.d: %.cpp
	$(CXX) -MM -MF $@ $<
 
%.o: %.d
 
-include $(OBJS:.o=.d)
