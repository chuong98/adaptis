# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# # %%
# from IPython import get_ipython

# %%
# get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

import cv2
import sys
import torch
from tqdm.notebook import tnrange

# sys.path.insert(0, '..')
from adaptis.inference.adaptis_sampling import get_panoptic_segmentation
from adaptis.inference.prediction_model import AdaptISPrediction

device = torch.device('cuda:0')

# %% [markdown]
# ### Init dataset

# %%
from adaptis.data.toy import ToyDataset

dataset_path = '/data/adaptis_toy/augmented'
dataset = ToyDataset(dataset_path, split='test', with_segmentation=True)

# %% [markdown]
# ### Load model

# %%
from adaptis.model.toy.models import get_unet_model

model = get_unet_model(norm_layer=torch.nn.BatchNorm2d, with_proposals=True)
pmodel = AdaptISPrediction(model, dataset, device)

weights_path = './experiments/toy_unet/checkpoints/toy_proposals_last_checkpoint.params'
pmodel.load_parameters(weights_path)


# %%
dir(pmodel)
print(pmodel.net)

# %% [markdown]
# ### Define evaluation function

# %%
from adaptis.coco.panoptic_metric import PQStat, pq_compute, print_pq_stat


def test_model(pmodel, dataset,
               sampling_algorithm, sampling_params,
               use_flip=False, cut_radius=-1):
    pq_stat = PQStat()
    categories = dataset._generate_coco_categories()
    categories = {x['id']: x for x in categories}

    for indx in tnrange(len(dataset)):
        sample = dataset.get_sample(indx)
        pred = get_panoptic_segmentation(pmodel, sample['image'],
                                         sampling_algorithm=sampling_algorithm,
                                         use_flip=use_flip, cut_radius=cut_radius, **sampling_params)


        coco_sample = dataset.convert_to_coco_format(sample)
        pred = dataset.convert_to_coco_format(pred)

        pq_stat = pq_compute(pq_stat, pred, coco_sample, categories)

    print_pq_stat(pq_stat, categories)

# %% [markdown]
# ### Test proposals-based point sampling

# %%
proposals_sampling_params = {
    'thresh1': 0.4,
    'thresh2': 0.5,
    'ithresh': 0.3,
    'fl_prob': 0.10,
    'fl_eps': 0.003,
    'fl_blur': 2,
    'max_iters': 100
}

test_model(pmodel, dataset,
           sampling_algorithm='proposals',
           sampling_params=proposals_sampling_params,
           use_flip=False)


# %%
proposals_sampling_params = {
    'thresh1': 0.4,
    'thresh2': 0.5,
    'ithresh': 0.3,
    'fl_prob': 0.10,
    'fl_eps': 0.003,
    'fl_blur': 2,
    'max_iters': 100
}

test_model(pmodel, dataset,
           sampling_algorithm='proposals',
           sampling_params=proposals_sampling_params,
           use_flip=True)

# %% [markdown]
# ### Test random sampling

# %%
random_sampling_params = {
    'thresh1': 0.4,
    'thresh2': 0.5,
    'ithresh': 0.3,
    'num_candidates': 7,
    'num_iters': 40
}

test_model(pmodel, dataset,
           sampling_algorithm='random', sampling_params=random_sampling_params,
           use_flip=False)


# %%
random_sampling_params = {
    'thresh1': 0.4,
    'thresh2': 0.5,
    'ithresh': 0.3,
    'num_candidates': 7,
    'num_iters': 40
}

import pdb; pdb.set_trace()
test_model(pmodel, dataset,
           sampling_algorithm='random', sampling_params=random_sampling_params,
           use_flip=True)

# %% [markdown]
# ### Results visualization

# %%
from adaptis.utils.vis import visualize_instances, visualize_proposals


proposals_sampling_params = {
    'thresh1': 0.5,
    'thresh2': 0.5,
    'ithresh': 0.3,
    'fl_prob': 0.10,
    'fl_eps': 0.003,
    'fl_blur': 2,
    'max_iters': 100
}

vis_samples = [15, 25, 42]


fig, ax = plt.subplots(nrows=len(vis_samples), ncols=3, figsize=(7,7))
fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)

for row_indx, sample_indx in enumerate(vis_samples):
    sample = dataset.get_sample(sample_indx)
    pred = get_panoptic_segmentation(pmodel, sample['image'],
                                 sampling_algorithm='proposals',
                                 use_flip=True, **proposals_sampling_params)

    for i in range(3):
        ax[row_indx, i].axis('off')

    if row_indx == 0:
        ax[row_indx, 0].set_title('Input Image', fontsize=14)
        ax[row_indx, 1].set_title('Instance Segmentation', fontsize=14)
        ax[row_indx, 2].set_title('Proposal Map', fontsize=14)
    ax[row_indx, 0].imshow(sample['image'])
    ax[row_indx, 1].imshow(visualize_instances(pred['instances_mask']))
    ax[row_indx, 2].imshow(visualize_proposals(pred['proposals_info']))

# %% [markdown]
# ### Test challenging samples

# %%
from adaptis.utils.vis import visualize_instances, visualize_proposals

dense_sampling_params = {
    'thresh1': 0.75,
    'thresh2': 0.50,
    'ithresh': 0.3,
    'fl_prob': 0.10,
    'fl_eps': 0.003,
    'fl_blur': 2,
    'max_iters': 1000,
    'cut_radius': 48
}

sample_image = cv2.imread('../images/toy_samples/250_sample_noisy.jpg')[:, :, ::-1].copy()
plt.figure(figsize=(8, 8))
plt.imshow(sample_image)

pred = get_panoptic_segmentation(pmodel, sample_image,
                                 sampling_algorithm='proposals',
                                 use_flip=True, **dense_sampling_params)

plt.figure(figsize=(8,8))
plt.imshow(visualize_instances(pred['instances_mask'],
                               boundaries_color=(150, 150, 150), boundaries_alpha=0.8))

plt.figure(figsize=(8,8))
plt.imshow(visualize_proposals(pred['proposals_info']))

