SRCDIR=src
INCDIR=include
OBJDIR=obj
BINDIR=bin
DATADIR=data
MODELSDIR=models

SRCFILES=$(wildcard ${SRCDIR}/*.cpp)
OBJFILES=$(patsubst ${SRCDIR}/%.cpp,${OBJDIR}/%.o, ${SRCFILES})

INCLUDE=-I${INCDIR} -I/usr/include/libxml++-2.6 -I/usr/lib/libxml++-2.6/include -I/usr/include/libxml2 -I/usr/include/glibmm-2.4 -I/usr/lib/glibmm-2.4/include -I/usr/include/sigc++-2.0 -I/usr/lib/sigc++-2.0/include -I/usr/include/glib-2.0 -I/usr/lib/glib-2.0/include
LIBS=-lgsl -lgslcblas -lm -pthread -lxml++-2.6 -lxml2 -lglibmm-2.4 -lgobject-2.0 -lsigc-2.0 -lgthread-2.0 -lrt -lglib-2.0
FLAGS=-std=c++0x -Wall -g -O3 `pkg-config --cflags glib-2.0`
GCC=g++

main: ${OBJFILES}
	${GCC} ${FLAGS} ${LIBS} -o ${BINDIR}/$@ ${OBJFILES}

${OBJDIR}/%.o: ${SRCDIR}/%.cpp ${INCDIR}/%.h 
	${GCC} ${FLAGS} ${INCLUDE} -c $< -o $@

.PHONY: clean

clean:
	rm -f ${OBJDIR}/* ${BINDIR}/*

