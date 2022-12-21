#do
#module load cuda/11.7
#module load nvhpc-no-mpi/22.5
#ulimit -s unlimited 
#or 
#source envs
F90       = nvfortran
ACCFLAGS  = -acc -ta=nvidia:managed -Minfo=accel
ACCFLAGSU = -acc -ta=nvidia -Minfo=accel
OMPFLAGS  = -mp -Minfo
OMPGPUFLAGS  = -ta=tesla,cc70,cuda11.7 -mp=gpu  -Minfo
FASTFLAGS = -O2

BIN = benchmark2d_omp benchmark2d benchmark1_omp 

all: $(BIN)

benchmark1_omp: benchmark1_omp.f90
	$(F90) $(FASTFLAGS) $(OMPGPUFLAGS) -o $@ $<

benchmark1_x100_omp: benchmark1_x100_omp.f90
	$(F90) $(FASTFLAGS) $(OMPGPUFLAGS) -o $@ $<

benchmark2d_omp.o: benchmark2d_omp.f90
	$(F90) $(FASTFLAGS) $(OMPGPUFLAGS) -c $<

benchmark2d:     benchmark2d_omp.o benchmark2d_driver.f90
	$(F90) $(FASTFLAGS) -mp $^ -o $@


clean:
	$(RM) $(BIN) *.o
