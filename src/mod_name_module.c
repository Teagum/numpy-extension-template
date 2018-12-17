#define NPY_NO_DEPRECATED_API NPY_1_8_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>


static PyObject *
mod_name_func1(PyObject *self, PyObject *args)
{
	/* Local representation of the input as Python Object */
	PyObject *py_x = NULL;

	/* Local representation as an array array */
	PyArrayObject *arr_x = NULL;

	/* Parse the argument list unsing specific format strings. There is, however,
	 * not format string for numpy arrays. That's why we need to parse the input
	 * as an generic PythonObject using the "O" format string.*/
	if (!PyArg_ParseTuple(args, "O", &py_x))
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
	arr_x = (PyArrayObject *)PyArray_FROM_OTF(py_x, NPY_LONGDOUBLE, NPY_ARRAY_IN_ARRAY);
	if (arr_x == NULL) return NULL;

	/* get array attributes, e.g., element count ... */
	int		n		= (int) PyArray_SIZE(arr_x);

	/* Obtain a pointer to the data buffer of the array */
	double	*data	= (double *) PyArray_DATA(arr_x);

	/* 
	 * do your stuff
	 */

	/* Clean up your array objects */
	Py_DECREF(arr_x);

	/* Build return values */
	PyObject *ret = Py_BuildValue("i", 1);
	return ret;
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
