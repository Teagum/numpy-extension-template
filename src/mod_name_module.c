#define NPY_NO_DEPRECATED_API NPY_1_8_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>


static PyObject *
mod_name_func1(PyObject *self, PyObject *args)
{
	/* Local representation of the args that come from Python */
	PyObject *X_from_python = NULL;

	/* Parse the argument list unsing specific format strings. There is, however,
	 * not format string for numpy arrays. That's why we need to parse the input
	 * as an generic PythonObject using the "O" format string.*/
	if (!PyArg_ParseTuple (args, "O", &X_from_python))
		return NULL;

	/* As PyArg_ParseTuple returns a PyObject, we have to reinterpret it as
	 * a PyArrayObject. One way to do so is by using PyArray_FROM_OTF, where "OTF"
	 * stands for "Object Type Flags" (so I guess). It takes an PyOject, an enumerated
	 * numpy type, and some format flags. This allows to interprete any iterable Python 
	 * object as numpy array of the given format. Note: Py_Array_FROM_OTF, as well as all
	 * other macros that fall back to PyArray_FromAny, steals a reference to the underlying 
	 * descriptor. That means you own a references to the descriptor and, hence, your are
	 * responsible decrease the descriptors reference count by using Py_DECREF on your arry
	 * pointer
	 */
	PyArrayObject *X_numpy = (PyArrayObject *) PyArray_FROM_OTF (X_from_python, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
	if (X_numpy == NULL) return NULL;

	/* get array attributes, e.g., element count ... */
	npy_intp N = PyArray_SIZE(X_numpy);

	/* Obtain a pointer to the data buffer of the array */
	double *X_np_data = (double *) PyArray_DATA (X_numpy);

	/* Print out data from Python */
	puts("\nData received from Python:");
	for (npy_intp i = 0; i < N; i++)
	{
		fprintf(stdout, "%5.3f\t", X_np_data[i]);
		fflush(stdout);
	}

	/* Dynamically allocate memory for a C array just for the sake of it */
	double *X_carray = malloc (N * sizeof (double));
	if (X_carray == NULL) return NULL;
	
	for (npy_intp i = 0; i < N; i++)
		X_carray[i] = (double) i * i;

	/* Create an array wrapper around the allocated memory.
	 * dims sets the number of elements in each of nd dimensions.  */
	int nd = 2;
	npy_intp dims[] = {2, 3};
	PyObject *out = PyArray_SimpleNewFromData (nd, dims, NPY_DOUBLE, X_carray);

	double *X_ca_data = PyArray_DATA ((PyArrayObject *) out);
	puts("\nData int wrapper:");
	for (npy_intp i = 0; i < N; i++)
	{
		fprintf(stdout, "%5.3f\t", X_ca_data[i]);
		fflush(stdout);
	}
	/* 
	 * do some more stuff
	 */

	/* Clean up your array objects */

	/* Build return values */
	PyObject *val = Py_BuildValue ("O", out);

	free(X_carray);
	Py_DECREF (X_numpy);
	return val;
}


static PyMethodDef
mod_name_methods[] = {
	{"func1", mod_name_func1, METH_VARARGS, "func1(x, y, z)"},
	{NULL, NULL, 0, NULL}
};


static PyModuleDef
mod_name_module = { 
	PyModuleDef_HEAD_INIT, "mod_name", NULL, -1, mod_name_methods
};


PyMODINIT_FUNC
PyInit_mod_name(void)
{
	import_array();
	return PyModule_Create (&mod_name_module);
}
