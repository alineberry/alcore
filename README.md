# Adam Lineberry's general purpose code
This is the supporting code for any task. Includes general utility functions and connection classes for Hive and Postgress

# HOWTO build

## On *nix
```bash
make test
make lint
make clean
```

## On Windows
```bash
Makefile.bat test
Makefile.bat lint
Makefile.bat clean
```

# HOWTO create an egg
```bash
python setup.py bdist_egg
easy_install dist/<current_egg_filename>.egg
```

Or better yet, in one line `python setup.py install`.
