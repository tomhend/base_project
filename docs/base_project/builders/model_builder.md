Module base_project.builders.model_builder
==========================================
Builder file for models, the build_model function is called from 'run_manager'
to create the model.

Functions
---------

    
`build_model(model_name: str, **kwargs: dict[str, any]) -> torch.nn.modules.module.Module`
:   Main function for building the model. Uses the model_name to select the correct builder
    function from MODEL_CONSTRUCTORS, and passes the kwargs on to this function
    
    Args:
        model_name (str): name of the model, should be in MODEL_CONSTRUCTORS
        kwargs (dict[str, any]): the keyword arguments to be passed onto the builder function
    
    Returns:
        torch.nn.Module: returns the built model

    
`test_resnet(**kwargs: dict[str, any]) -> torch.nn.modules.module.Module`
:   Builds a torchvision resnet18 with the given kwargs
    
    Args:
        kwargs (dict[str, any]):
            see https://pytorch.org/vision/stable/models/resnet.html for available kwargs
    
    Returns:
        torch.nn.Module: the created resnet18