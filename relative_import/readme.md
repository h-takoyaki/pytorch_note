**main trick**:

add the path of `lib` into `sys.path`

~~~python
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, str(path))
~~~

**file structure**:

~~~shell
.
├───lib
│   ├───core
│   │   │   core_function.py
│   │   │   __init__.py
│   │   │
│   │   ├───subcore
│   │   │   │   subcore_function.py
│   │
│   └───models
│       │   model_x.py
│       │   __init__.py
│
└───tools
    │   train.py
    │   _init_path.py

~~~

> **Just knowing what directory a file is in does not determine what package Python thinks it is in.** 
>
> When a file is loaded, it is given a name (which is stored in its `__name__` attribute).
>
> - If it was loaded as the top-level script, its name is `__main__`.
> - If it was loaded as a module, its name is [ the filename, preceded by the names of any packages/subpackages of which it is a part, separated by dots ], for example, `package.subpackage1.moduleX`.
>
> [from **Script vs. Module**](https://stackoverflow.com/a/14132912/15554546)

**Solutions:**

1.  `python -m package.subpackage1.moduleX`. The `-m` tells Python to load it as a module, not as the top-level script.
2. put `myfile.py` *somewhere else* – *not* inside the `package` directory – and run it. If inside `myfile.py` you do things like `from package.moduleA import spam`, it will work fine.
3. :thumbsup:[researchmm/2D-TAN](https://github.com/researchmm/2D-TAN) add `./tools/../lib` into `sys.path`

## reference

[researchmm/2D-TAN](https://github.com/researchmm/2D-TAN)

[**Script vs. Module**](https://stackoverflow.com/a/14132912/15554546)