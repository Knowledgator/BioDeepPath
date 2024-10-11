import os

import torch
import torchkge
from torch import optim
from torchkge import MarginLoss, TransEModel
from torchkge.sampling import BernoulliNegativeSampler
from torchkge.utils import DataLoader, load_fb15k
from tqdm.auto import tqdm


def train_transE_model_on_freebase15k(
    embed_dim=100,
    lr=0.0004,
    epochs=1000,
    batch_size=32768,
    margin=0.5,
    normalize_after_training=True,
):
    kg_train, _, _ = load_fb15k()
    model = TransEModel(
        embed_dim, kg_train.n_ent, kg_train.n_rel, dissimilarity_type="L2"
    )
    criterion = MarginLoss(margin)


    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        model.cuda()
        criterion.cuda()

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                     factor=0.1,
                                                     patience=5)

    sampler = BernoulliNegativeSampler(kg_train)
    dataloader = DataLoader(kg_train, batch_size=batch_size, use_cuda="all")

    iterator = tqdm(range(epochs), unit="epoch")
    for epoch in iterator:
        running_loss = 0.0
        for i, batch in enumerate(dataloader):
            h, t, r = batch[0], batch[1], batch[2]
            n_h, n_t = sampler.corrupt_batch(h, t, r)

            optimizer.zero_grad()

            # forward + backward + optimize
            pos, neg = model(h, t, r, n_h, n_t)
            loss = criterion(pos, neg)
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
        iterator.set_description(
            "Epoch {} | mean loss: {:.5f}".format(
                epoch + 1, running_loss / len(dataloader)
            )
        )

    torch.save(model.state_dict(), "trans_e_model_weights.pt")
    if normalize_after_training:
        model.normalize_parameters()

    return model


def train_transE_model(
    dataset: torchkge.data_structures.KnowledgeGraph,
    embed_dim=100,
    lr=0.0004,
    epochs=1000,
    batch_size=32768,
    margin=0.5,
    normalize_after_training=True,
    save_dir=None,
    model_name='transE_weights.pt'
):
    model = TransEModel(
        embed_dim, dataset.n_ent, dataset.n_rel, dissimilarity_type="L2"
    )
    criterion = MarginLoss(margin)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        model.cuda()
        criterion.cuda()

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    sampler = BernoulliNegativeSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=batch_size, use_cuda="all")

    iterator = tqdm(range(epochs), unit="epoch")
    for epoch in iterator:
        running_loss = 0.0
        for i, batch in enumerate(dataloader):
            h, t, r = batch[0], batch[1], batch[2]
            n_h, n_t = sampler.corrupt_batch(h, t, r)

            optimizer.zero_grad()

            # forward + backward + optimize
            pos, neg = model(h, t, r, n_h, n_t)
            loss = criterion(pos, neg)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        iterator.set_description(
            "Epoch {} | mean loss: {:.5f}".format(
                epoch + 1, running_loss / len(dataloader)
            )
        )

    if save_dir is None:
        save_dir = model_name
    else:
        save_dir = os.path.join(save_dir, model_name)

    torch.save(model.state_dict(), save_dir)
    if normalize_after_training:
        model.normalize_parameters()

    return model
