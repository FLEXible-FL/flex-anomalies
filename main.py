from main_static import hub, model_tests
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.description = "CLI application for outlier detection experiments in a federated environment, we can handle two options for experiment params, a json file with all params is preferred and the output will take the same name as the experiment file"
    parser.add_argument("--json_file", help="Path to file with experiments params")
    parser.add_argument(
        "--model",
        help=f"Model name to run experiment one of: {list(model_tests.keys())}",
    )
    parser.add_argument(
        "--model_params", help="Inline model pararms in json string format"
    )
    parser.add_argument("--dataset", help="Path to dataset to use in the experiment")
    parser.add_argument(
        "--split_size", help="Float value indicating the split size for test"
    )
    parser.add_argument(
        "--n_clients",
        help="Number of clients to split the data for federated experiments",
    )
    parser.add_argument(
        "--n_rounds",
        help="Number of rounds of model training in the clients for federatd experiments",
    )
    parser.add_argument("--output_name", help="Output name of experiments results")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args = {k:v for k,v in args.__dict__.items() if v is not None}
    print(args)
    hub(**args)

    