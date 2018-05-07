import argparse


def get_params(arguments):
    parser = argparse.ArgumentParser(prog="hyperparams",
                                     usage="%(prog)s [options]")
    parser.add_argument("--train-files", required=True)
    parser.add_argument("--job-dir", required=True)
    parser.add_argument("--total_size", required=True, type=int)
    parser.add_argument("--n_timesteps", type=int, default=30)
    parser.add_argument("--n_neurons", type=int, default=128)
    parser.add_argument("--n_outputs", type=int, default=10)
    parser.add_argument("--n_inputs_per_timestep", type=int, default=1)
    parser.add_argument("--n_hidden", type=int, default=100)
    parser.add_argument("--n_steps", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--cell_type", type=str,
                        choices=["gru", "lstm", "rnn", "fused_lstm"],
                        default="gru")

    args = parser.parse_args(arguments)
    return vars(args)


if __name__ == "__main__":
    import sys
    print get_params(sys.argv[1:])
