import torchinfo

_MODELS = dict()


def register(name):
    def add_to_dict(fn):
        global _MODELS
        _MODELS[name] = fn
        return fn

    return add_to_dict


def get_model(args, summary=None):
    """Initialize model"""
    assert args.model in _MODELS.keys(), f"model {args.model} not found."

    # model should output logits with BCE loss
    args.output_logits = (
        args.loss in ["bce", "binarycrossentropy"] and args.model != "identity"
    )

    model = _MODELS[args.model](args)
    model.to(args.device)

    if summary is not None:
        # get model summary and write to args.output_dir
        model_info = torchinfo.summary(
            model,
            input_size=(args.batch_size, *args.input_shape),
            device=args.device,
            verbose=0,
        )
        with open(args.output_dir / "model.txt", "w") as file:
            file.write(str(model_info))
        summary.scalar("model/trainable_parameters", model_info.trainable_params)
        if args.verbose == 2:
            print(str(model_info))

    return model
