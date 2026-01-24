import time

from openpi.policies import libero_policy
from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config


def main():
    # config = _config.get_config("pi0_fast_libero")
    # checkpoint_dir = download.maybe_download("s3://openpi-assets/checkpoints/pi0_fast_libero")

    config = _config.get_config("pi0_libero")
    checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi0_libero")

    # Create a trained policy.
    policy = _policy_config.create_trained_policy(config, checkpoint_dir)

    for n_samples in [1, 10, 20]:
        print()
        print("#" * 20)
        print(f"Testing with {n_samples} samples...")
        policy._sample_kwargs["temperature"] = 0.5
        policy._sample_kwargs["n_action_samples"] = n_samples

        policy_recorder = _policy.PolicyRecorder(policy, "rollouts/debug")

        print("Starting the first inference...")
        starting_time = time.time()
        example = libero_policy.make_libero_example()
        result = policy_recorder.infer(example)
        print("Actions shape:", result["actions"].shape)
        print("Time taken:", time.time() - starting_time)

        print("Run inference 10 times...")
        starting_time = time.time()
        for i in range(10):
            example = libero_policy.make_libero_example()
            result = policy_recorder.infer(example)

        print("Actions shape:", result["actions"].shape)
        print("Time taken:", time.time() - starting_time)


if __name__ == "__main__":
    main()
