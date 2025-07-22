import wandb

wandb.init(project="my-awesome-project", name="run-1")

for epoch in range(10):
    loss = 1.0
    wandb.log({"loss": loss, "epoch": epoch})

wandb.init(config={"lr": 0.001, "batch_size": 32})
