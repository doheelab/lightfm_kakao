import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal

import pytest

import scipy.sparse as sp

from lightfm import LightFM
import lightfm
import lightfm.model
import lightfm.inference as inference

mattypes = sp.coo_matrix, sp.lil_matrix, sp.csr_matrix, sp.csc_matrix
dtypes = np.int32, np.int64, np.float32, np.float64


def test_empty_matrix():
    no_users, no_items = 10, 100
    train = sp.coo_matrix((no_users, no_items), dtype=np.int32)
    model = LightFM()
    model.fit_partial(train)


def test_matrix_types():
    no_users, no_items = 10, 100
    no_features = 20

    for mattype in mattypes:
        for dtype in dtypes:
            train = mattype((no_users, no_items), dtype=dtype)

            user_features = mattype((no_users, no_features), dtype=dtype)
            item_features = mattype((no_items, no_features), dtype=dtype)

            model = LightFM()
            model.fit_partial(train, user_features=user_features, item_features=item_features)

            model.predict(
                np.random.randint(0, no_users, 10).astype(np.int32),
                np.random.randint(0, no_items, 10).astype(np.int32),
                user_features=user_features,
                item_features=item_features,
            )

            model.predict_rank(train, user_features=user_features, item_features=item_features)


def test_coo_with_duplicate_entries():
    # Calling .tocsr on a COO matrix with duplicate entries
    # changes its data arrays in-place, leading to out-of-bounds
    # array accesses in the WARP code.
    # Reported in https://github.com/lyst/lightfm/issues/117.

    rows, cols = 1000, 100
    mat = sp.random(rows, cols)
    mat.data[:] = 1

    # Duplicate entries in the COO matrix
    mat.data = np.concatenate((mat.data, mat.data[:1000]))
    mat.row = np.concatenate((mat.row, mat.row[:1000]))
    mat.col = np.concatenate((mat.col, mat.col[:1000]))

    for loss in ('warp', 'bpr', 'warp-kos'):
        model = LightFM(loss=loss)
        model.fit(mat)


def test_predict():
    no_users, no_items = 10, 100

    train = sp.coo_matrix((no_users, no_items), dtype=np.int32)

    model = LightFM()
    model.fit_partial(train)

    for uid in range(no_users):
        scores_arr = model.predict(np.repeat(uid, no_items), np.arange(no_items))
        scores_int = model.predict(uid, np.arange(no_items))
        assert np.allclose(scores_arr, scores_int)


def test_input_dtypes():
    no_users, no_items = 10, 100
    no_features = 20

    for dtype in dtypes:
        train = sp.coo_matrix((no_users, no_items), dtype=dtype)
        user_features = sp.coo_matrix((no_users, no_features), dtype=dtype)
        item_features = sp.coo_matrix((no_items, no_features), dtype=dtype)

        model = LightFM()
        model.fit_partial(train, user_features=user_features, item_features=item_features)

        model.predict(
            np.random.randint(0, no_users, 10).astype(np.int32),
            np.random.randint(0, no_items, 10).astype(np.int32),
            user_features=user_features,
            item_features=item_features,
        )


def test_not_enough_features_fails():
    no_users, no_items = 10, 100
    no_features = 20

    train = sp.coo_matrix((no_users, no_items), dtype=np.int32)

    user_features = sp.csr_matrix((no_users - 1, no_features), dtype=np.int32)
    item_features = sp.csr_matrix((no_items - 1, no_features), dtype=np.int32)
    model = LightFM()
    with pytest.raises(Exception):
        model.fit_partial(train, user_features=user_features, item_features=item_features)


def test_feature_inference_fails():
    # On predict if we try to use feature inference and supply
    # higher ids than the number of features that were supplied to fit
    # we should complain

    no_users, no_items = 10, 100
    no_features = 20

    train = sp.coo_matrix((no_users, no_items), dtype=np.int32)

    user_features = sp.csr_matrix((no_users, no_features), dtype=np.int32)
    item_features = sp.csr_matrix((no_items, no_features), dtype=np.int32)
    model = LightFM()
    model.fit_partial(train, user_features=user_features, item_features=item_features)

    with pytest.raises(ValueError):
        model.predict(np.array([no_features], dtype=np.int32), np.array([no_features], dtype=np.int32))


def test_return_self():
    no_users, no_items = 10, 100

    train = sp.coo_matrix((no_users, no_items), dtype=np.int32)
    model = LightFM()
    assert model.fit_partial(train) is model
    assert model.fit(train) is model


def test_param_sanity():

    with pytest.raises(AssertionError):
        LightFM(no_components=-1)

    with pytest.raises(AssertionError):
        LightFM(user_alpha=-1.0)

    with pytest.raises(AssertionError):
        LightFM(item_alpha=-1.0)

    with pytest.raises(ValueError):
        LightFM(max_sampled=-1.0)


def test_sample_weight():
    model = LightFM()

    train = sp.coo_matrix(np.array([[0, 1], [0, 1]]))

    with pytest.raises(ValueError):
        # Wrong number of weights
        sample_weight = sp.coo_matrix(np.zeros((2, 2)))

        model.fit(train, sample_weight=sample_weight)

    with pytest.raises(ValueError):
        # Wrong shape
        sample_weight = sp.coo_matrix(np.zeros(2))
        model.fit(train, sample_weight=sample_weight)

    with pytest.raises(ValueError):
        # Wrong order of entries
        model.fit(train, sample_weight=sample_weight)

    sample_weight = sp.coo_matrix((train.data, (train.row, train.col)))
    model.fit(train, sample_weight=sample_weight)

    model = LightFM(loss='warp-kos')

    with pytest.raises(NotImplementedError):
        model.fit(train, sample_weight=np.ones(1))


def test_predict_ranks():
    no_users, no_items = 10, 100

    train = sp.rand(no_users, no_items, format='csr', random_state=42)

    model = LightFM()
    model.fit_partial(train)

    # Compute ranks for all items
    rank_input = sp.csr_matrix(np.ones((no_users, no_items)))
    ranks = model.predict_rank(rank_input, num_threads=2).todense()

    assert np.all(ranks.min(axis=1) == 0)
    assert np.all(ranks.max(axis=1) == no_items - 1)

    for row in range(no_users):
        assert np.all(np.sort(ranks[row]) == np.arange(no_items))

    # Train set exclusions. All ranks should be zero
    # if train interactions is dense.
    ranks = model.predict_rank(rank_input,
                               train_interactions=rank_input).todense()
    assert np.all(ranks == 0)

    # Max rank should be num_items - 1 - number of positives
    # in train in that row
    ranks = model.predict_rank(rank_input,
                               train_interactions=train).todense()
    assert np.all(
        np.squeeze(np.array(ranks.max(axis=1))) == no_items - 1 - np.squeeze(np.array(train.getnnz(axis=1)))
    )

    # Make sure ranks are computed pessimistically when
    # there are ties (that is, equal predictions for every
    # item will assign maximum rank to each).
    model.user_embeddings = np.zeros_like(model.user_embeddings)
    model.item_embeddings = np.zeros_like(model.item_embeddings)
    model.user_biases = np.zeros_like(model.user_biases)
    model.item_biases = np.zeros_like(model.item_biases)

    ranks = model.predict_rank(rank_input, num_threads=2).todense()

    assert np.all(ranks.min(axis=1) == 99)
    assert np.all(ranks.max(axis=1) == 99)

    # Wrong input dimensions
    with pytest.raises(ValueError):
        model.predict_rank(sp.csr_matrix((5, 5)), num_threads=2)


def test_exception_on_divergence():
    no_users, no_items = 1000, 1000
    train = sp.rand(no_users, no_items, format='csr', random_state=42)
    model = LightFM(learning_rate=10000000.0, loss='warp')
    with pytest.raises(ValueError):
        model.fit(train, epochs=10)


def test_sklearn_api():
    model = LightFM()
    params = model.get_params()
    model2 = LightFM(**params)
    params2 = model2.get_params()
    assert params == params2
    model.set_params(**params)
    params['invalid_param'] = 666
    with pytest.raises(ValueError):
        model.set_params(**params)


def test_predict_not_fitted():
    model = LightFM()

    with pytest.raises(ValueError):
        model.predict(np.arange(10), np.arange(10))

    with pytest.raises(ValueError):
        model.predict_rank(1)

    with pytest.raises(ValueError):
        model.get_user_representations()

    with pytest.raises(ValueError):
        model.get_item_representations()


def test_nan_features():
    no_users, no_items = 1000, 1000
    train = sp.rand(no_users, no_items, format='csr', random_state=42)

    features = sp.identity(no_items)
    features.data *= np.nan

    model = LightFM(loss='warp')
    with pytest.raises(ValueError):
        model.fit(train, epochs=10, user_features=features, item_features=features)


def test_nan_interactions():
    no_users, no_items = 1000, 1000

    train = sp.rand(no_users, no_items, format='csr', random_state=42)
    train.data *= np.nan

    model = LightFM(loss='warp')

    with pytest.raises(ValueError):
        model.fit(train)


def test_precompute_representation():
    n_users = 10 ** 3
    n_user_features = 100
    no_component = 50
    user_features = sp.random(n_users, n_user_features, density=.1)
    feature_embeddings = np.random.uniform(size=(n_user_features, no_component))
    feature_biases = np.random.uniform(size=n_user_features)
    features = user_features

    representation, representation_biases = inference._precompute_representation(
        features,
        feature_embeddings,
        feature_biases,
    )
    assert representation.shape == (n_users, no_component)
    assert representation_biases.shape == (n_users, )


def test_batch_predict():
    no_components = 2
    ds = RandomDataset(density=1.0)

    model = LightFM(no_components=no_components)
    model.fit_partial(ds.train, user_features=ds.user_features, item_features=ds.item_features)

    model.batch_setup(
        item_chunks={0: ds.item_ids},
        user_features=ds.user_features,
        item_features=ds.item_features,
    )
    user_repr = inference._user_repr
    item_repr = inference._item_repr
    assert np.sum(user_repr)
    assert user_repr.shape == (ds.no_users, no_components)
    assert np.sum(item_repr)
    assert item_repr.shape == (no_components, ds.no_items)

    zeros = 0

    for uid in range(ds.no_users):

        original_scores = model.predict(
            np.repeat(uid, ds.no_items),
            np.arange(ds.no_items),
            user_features=ds.user_features,
            item_features=ds.item_features,
        )

        # Check scores
        _, batch_predicted_scores = model.predict_for_user(user_id=uid, top_k=0, item_ids=ds.item_ids)
        assert_array_almost_equal(original_scores, batch_predicted_scores)

        # Check ids
        original_ids = np.argsort(-original_scores)[:5]
        batch_ids, _ = model.predict_for_user(user_id=uid, top_k=5, item_ids=ds.item_ids)
        assert np.array_equal(original_ids, batch_ids)

        if np.sum(batch_predicted_scores) == 0:
            zeros += 1
    assert zeros < ds.no_users, 'predictions seems to be all zeros'


def test_batch_predict_with_items():
    no_components = 2
    ds = RandomDataset(density=1.0)

    model = LightFM(no_components=no_components)
    model.fit_partial(ds.train, user_features=ds.user_features, item_features=ds.item_features)
    model.batch_setup(item_chunks={0: ds.item_ids}, user_features=ds.user_features, item_features=ds.item_features)
    n_items = 10
    item_ids = np.random.choice(ds.item_ids, n_items)

    for uid in range(ds.no_users):

        original_scores = model.predict(
            np.repeat(uid, n_items),
            item_ids=item_ids,
            user_features=ds.user_features,
            item_features=ds.item_features,
        )

        # Check scores
        _, batch_predicted_scores = model.predict_for_user(user_id=uid, item_ids=item_ids, top_k=0)
        assert_array_almost_equal(original_scores, batch_predicted_scores)

        # Check ids
        original_ids = item_ids[np.argsort(-original_scores)[:5]]
        batch_ids, _ = model.predict_for_user(user_id=uid, item_ids=item_ids, top_k=5)
        assert_array_equal(original_ids, batch_ids)


def test_predict_for_user_with_items():
    no_components = 2
    ds = RandomDataset(no_items=5, no_users=2, density=1.)
    model = LightFM(no_components=no_components)
    model.fit_partial(ds.train, user_features=ds.user_features, item_features=ds.item_features)
    inference._batch_cleanup()

    with pytest.raises(EnvironmentError):
        model.predict_for_user(user_id=0, top_k=2, item_ids=np.arange(2))

    model.batch_setup(
        item_chunks={0: ds.item_ids},
        user_features=ds.user_features,
        item_features=ds.item_features,
    )

    for user_id in range(ds.no_users):
        scores = model.predict_for_user(
            user_id=user_id,
            top_k=2,
            item_ids=np.arange(2),
        )
        assert len(scores) == 2


def test_batch_predict_user_recs_per_user():
    no_components = 2
    ds = RandomDataset()

    model = LightFM(no_components=no_components)
    model.fit_partial(ds.train, user_features=ds.user_features, item_features=ds.item_features)
    model.batch_setup(
        item_chunks={0: ds.item_ids},
        user_features=ds.user_features,
        item_features=ds.item_features,
    )

    for uid in range(ds.no_users):
        rec_item_ids, rec_scores = model.predict_for_user(
            user_id=uid,
            top_k=5,
            item_ids=ds.item_ids,
        )
        assert len(rec_scores) == 5
        assert_array_almost_equal(rec_scores, -1 * np.sort(-1 * rec_scores))


def test_batch_predict_user_recs_per_user_wo_features():
    no_components = 2
    ds = RandomDataset()

    model = LightFM(no_components=no_components)
    model.fit_partial(ds.train)

    for uid in range(ds.no_users):
        rec_item_ids, rec_scores = model.predict_for_user(
            user_id=uid,
            top_k=5,
            item_ids=ds.item_ids,
        )
        assert len(rec_scores) == 5
        assert_array_almost_equal(rec_scores, -1 * np.sort(-1 * rec_scores))


class RandomDataset:

    def __init__(self,
                 no_users: int=5,
                 no_items: int=100,
                 no_features: int=3,
                 density=.3):

        self.no_users = no_users
        self.no_items = no_items
        self.no_features = no_features
        self.density = density
        self.item_ids = np.arange(self.no_items)
        self.user_features = sp.random(no_users, no_features, density=self.density, dtype=lightfm.CYTHON_DTYPE)
        self.item_features = sp.random(no_items, no_features, density=self.density, dtype=lightfm.CYTHON_DTYPE)
        self.train = sp.coo_matrix((no_users, no_items), dtype=np.int32)


def test_full_batch_predict():
    no_components = 2
    top_k = 5
    ds = RandomDataset()

    model = LightFM(no_components=no_components)
    model.fit_partial(ds.train, user_features=ds.user_features, item_features=ds.item_features)
    user_ids = [0, 1, 2]
    chunks = {0: ds.item_ids}

    # Single process
    model.batch_setup(item_chunks=chunks, user_features=ds.user_features, item_features=ds.item_features, n_process=1)
    recoms = model.batch_predict(
        user_ids=user_ids,
        chunk_id=0,
        top_k=top_k,
    )
    for user_id in user_ids:
        assert user_id in recoms
        assert len(recoms[user_id][0]) == top_k
    initial_recoms = recoms
    model.batch_cleanup()

    model.batch_setup(item_chunks=chunks, user_features=ds.user_features, item_features=ds.item_features, n_process=2)

    # Multiple processes
    recoms = model.batch_predict(
        user_ids=user_ids,
        chunk_id=0,
        top_k=top_k,
    )
    for user_id in user_ids:
        assert user_id in recoms
        assert_array_almost_equal(recoms[user_id], initial_recoms[user_id])


def test_full_batch_predict_wo_features():
    no_components = 2
    top_k = 5
    ds = RandomDataset(density=1.0)

    model = LightFM(no_components=no_components)
    model.fit_partial(ds.train)
    user_ids = [0, 1, 2]

    # Single process
    model.batch_setup({0: ds.item_ids})
    recoms = model.batch_predict(
        user_ids=user_ids,
        chunk_id=0,
        top_k=top_k,
    )
    for user_id in user_ids:
        assert user_id in recoms
        assert len(recoms[user_id][0]) == top_k


def test_regression_full_batch_predict():
    no_components = 2
    np.random.seed(42)
    ds = RandomDataset(no_items=5, density=1)

    model = LightFM(no_components=no_components)
    model.fit(ds.train, user_features=ds.user_features, item_features=ds.item_features)

    # Set non zero biases
    model.item_biases += 0.2
    model.user_biases += 0.5
    user_ids = [0, 1, 2]

    model.batch_setup(item_chunks={0: ds.item_ids}, item_features=ds.item_features, user_features=ds.user_features)
    recoms = model.batch_predict(
        user_ids=user_ids,
        chunk_id=0,
        top_k=0,  # Score all items
    )
    zeros = 0
    for user_id in user_ids:
        scores = model.predict(
            user_ids=user_id,
            item_ids=ds.item_ids,
            item_features=ds.item_features,
            user_features=ds.user_features,
            num_threads=1,
        )
        if sum(scores) != 0:
            zeros += 1
        assert_array_almost_equal(recoms[user_id][1], scores)
    assert zeros != 0


def test_get_top_k_scores():
    scores = np.array([.2, .1, .05, .9])
    item_ids = np.arange(len(scores))

    # Without trimming to top k
    item_ids, new_scores = inference._get_top_k_scores(scores=scores, k=0, item_ids=item_ids)
    assert_array_almost_equal(new_scores, scores)
    assert_array_equal(item_ids, np.arange(4))

    # With trimming to top k
    item_ids, new_scores = inference._get_top_k_scores(scores=scores, k=2, item_ids=item_ids)
    assert_array_almost_equal(new_scores, np.array([.9, .2]))
    assert_array_equal(item_ids, np.array([3, 0]))

    # Check, that we returned original item ids, not indices
    items_to_recommend = np.array([0, 10, 20, 30])
    item_ids, new_scores = inference._get_top_k_scores(scores=scores, k=2, item_ids=items_to_recommend)
    assert_array_almost_equal(new_scores, np.array([.9, .2]))
    assert_array_equal(item_ids, np.array([30, 0]))
