Module base_project.settings_utils.utils
========================================
File containing some utility functions

Functions
---------

    
`parse_cfg(cfg_path: pathlib.Path) -> dict[any, any]`
:   Parses the yaml file to a dictionary
    
    Args:
        cfg_path (Path): Path instance that points to the configuration yaml file
    
    Returns:
        dict[any, any]: parsed yaml file

    
`time_func(func: Callable) -> Callable`
:   Decorator that times the execution time of a function or method
    
    Args:
        func (Callable): the function or method to be decorated
    
    Returns:
        Callable: the decorated function