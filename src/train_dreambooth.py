import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="DreamBooth fine-tuning entry point")
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--instance_data_dir", type=str, required=True)
    parser.add_argument("--class_data_dir", type=str)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--instance_prompt", type=str, required=True)
    parser.add_argument("--class_prompt", type=str)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--max_train_steps", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--prior_loss_weight", type=float, default=1.0)
    parser.add_argument("--mixed_precision", type=str, default="fp16")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--with_prior_preservation", action="store_true")
    parser.add_argument("--prior_generation_prompt", type=str)
    return parser.parse_args()


def main():
    parse_args()
    raise NotImplementedError("DreamBooth training pipeline will be implemented by Khang Hy")


if __name__ == "__main__":
    main()

