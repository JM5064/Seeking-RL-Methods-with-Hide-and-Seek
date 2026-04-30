import torch 
from stable_baselines3 import PPO


def load(name):
    model = PPO.load("hider_policy_"+name)

    dummy_input = torch.randn(1, 3, 32, 32).to(model.device)

    torch.onnx.export(
        model.policy,
        dummy_input,
        f"hider_brain_{name}.onx",
        opset_version=12,
        input_names=["obs_0"], 
        output_names=["output"],
        dynamic_axes={'obs_0': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )

    print("Model exported")