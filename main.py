from utils.helpers import init_args
from train.text_trainer import train_text_mode
from train.image_trainer import train_image_mode


def main():
    args = init_args()

    if args.modality == 'text':
        train_text_mode(args)
    elif args.modality == 'image':
        train_image_mode(args)
    else:
        raise ValueError("Modality must be either 'text' or 'image'")


if __name__ == "__main__":
    main()
