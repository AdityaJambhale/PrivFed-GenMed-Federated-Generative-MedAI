import flwr as fl

def fit_config(rnd: int):
    return {
        "round": rnd,
        "num_rounds": 3  # or whatever your total is
    }

fl.server.start_server(
    server_address="localhost:8080",
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=fl.server.strategy.FedAvg(
        on_fit_config_fn=fit_config
    )
)
