Module base_project.builders.optimizer_builder
==============================================
Builder file for optimizers, the build_optimizer function is called from 'run_manager'
to create the optimizer.

Functions
---------

    
`adam(model_parameters, **kwargs) -> torch.optim.adam.Adam`
:   Builds the torch Adam optimizer with the specifications found in the kwargs.
    
    Args:
        model_parameters (Generator[torch.Tensor, None, None]): the parameters that should be
            optimized, taken from the model with model.parameters()
        kwargs(dict[str, any]): see: https://pytorch.org/docs/stable/generated/torch.optim.Adam.html
        for available kwargs
    
    Returns:
        torch.optim.Adam: and instance of torch.optim.Adam initialized with the given kwargs

    
`build_optimizer(optimizer_name: str, model_parameters: Generator[torch.Tensor, NoneType, NoneType], **kwargs) -> torch.optim.optimizer.Optimizer`
:   Main function for building the optimizer. Uses the optimizer_name to select the correct builder
    function from OPTIMIZER_CONSTRUCTORS, and passes the model_parameters and kwargs on to this
    function
    
    Args:
        optimizer_name (str): name of the optimizer, should be in OPTIMIZER_CONSTRUCTORS
        model_parameters (Generator[torch.Tensor, None, None]): the parameters that should be
            optimized, taken from the model with model.parameters()
        kwargs (dict[str, any]): the keyword arguments to be passed onto the builder function
    
    Returns:
        torch.optim.Optimizer: returns the built optimizer