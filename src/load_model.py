from .make_model import make_model


def load_model(c, device):
    models = []

    for training in c.params.pretrained:
        c.params.model = training.model
        model = make_model(c, device, training.dir)
        models.append(model)

    return models
