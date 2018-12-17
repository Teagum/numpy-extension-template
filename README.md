# numpy-extension-template
Basic template to write numpy extension modules for Python3.

## 1. Write some C function
Write some function in C and put it in `utilities.c`, or create a new file. If your create a new file for your custom function, remember to append the path to it to the `sources` list in `setup.py`.

## 2. Python interface
For each C function you want to expose to Python, you need to write an interface. An interface template can be found in `src/mod_name_module.c`.

## 3. Compile your extension module
Go to the root folder of your extension module and run

    python3 setup.py build_ext
    
This command automatically compiles the extension module and places it under the `build/` directory.
If you wish to have your new module in the my directory, run

    python3 setup.py build_ext --inplace
