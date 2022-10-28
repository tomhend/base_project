from trainers.base_trainer import BaseTrainer

def build_trainer(trainer_name: str, model, loss_fn, optimizer, device, **kwargs) -> BaseTrainer:
    trainer = TRAINER_CONSTRUCTORS[trainer_name](model, loss_fn, optimizer, device, **kwargs)
    return trainer
    
def base_trainer(model, loss_fn, optimizer, device, **kwargs) -> BaseTrainer:
    trainer = BaseTrainer(model, loss_fn, optimizer, device)
    return trainer

TRAINER_CONSTRUCTORS = {
    'base_trainer': base_trainer
}
