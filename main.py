from read_data import read_params, get_data_loaders
import tensorflow as tf
from Simulator import simulator
from pathlib import Path
import os

print(tf.__version__)

if __name__ == "__main__":

    directory = "Results"
    parent_dir = Path(__file__).parent.resolve()
    path = os.path.join(parent_dir, directory)
    try:
        os.mkdir(path)
    except OSError as error:
        print("Directory already created")

    current_folder = Path(__file__).parent.resolve()
    params_path = current_folder.joinpath('./params.yaml')

    params = read_params(params_path)
    n_nodes = params.get('simulation').get('n_clients')
    dataset = params.get('data').get('dataset')
    n_runs = params.get('simulation').get('tot_sims')

    for run in range(n_runs):

        client_train_loaders, server_test_loader, shape, num_classes, fracs, samples_users = get_data_loaders(
            dataset=dataset,
            nodes=n_nodes)

        simulator = simulator(params,
                              train_data=client_train_loaders,
                              test_data=server_test_loader,
                              shape=shape,
                              num_classes=num_classes,
                              fracs=fracs,
                              samples_user=samples_users,
                              run=run)

        simulator.start()

