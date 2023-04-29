from model_setups import baseline, baseline_weighted
import util.local_config as local_config
import wandb

def train_model_setup(setup_id):
    if setup_id == 'baseline':
        print('Training baseline setup')
        baseline.run()
    elif setup_id == 'baseline_weighted':
        print('Training baseline weighted setup')
        baseline_weighted.run()
    else:
        raise KeyError(f'Model setup: \'{setup_id}\' not found')

if __name__ == '__main__':
    # Set the setup id
    setup_id = 'baseline_weighted'

    # Login in wandb
    wandb.login()

    # Train the setup
    train_model_setup(setup_id=setup_id)