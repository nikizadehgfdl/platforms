/*
compiled/ran on gfdl_ws via
module load gcc/9.2.0
module load python/3.7.7
gcc -I/app/spack/v0.15/linux-rhel7-x86_64/gcc-4.8.5/python/3.7.7-d6cyi6ophaei6arnmzya2kn6yumye2yl/include/python3.7m -L/app/spack/v0.15/linux-rhel7-x86_64/gcc-4.8.5/python/3.7.7-d6cyi6ophaei6arnmzya2kn6yumye2yl/lib -lpython3.7m test_phyton_embedding.cc
./a.out
*/

#define PY_SSIZE_T_CLEAN
#include <Python.h>

int
main(int argc, char *argv[])
{
    wchar_t *program = Py_DecodeLocale(argv[0], NULL);
    if (program == NULL) {
        fprintf(stderr, "Fatal error: cannot decode argv[0]\n");
        exit(1);
    }
    Py_SetProgramName(program);  /* optional but recommended */
    Py_Initialize();
    PyRun_SimpleString("from time import time,ctime\n"
                       "print('Today is', ctime(time()))\n");
    if (Py_FinalizeEx() < 0) {
        exit(120);
    }
    PyMem_RawFree(program);
    return 0;
}
