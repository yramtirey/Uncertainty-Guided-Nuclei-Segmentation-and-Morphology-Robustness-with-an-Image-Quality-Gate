# ============================================================
# project to-do / implementation guide
# updated based on 03/19 meeting notes + original proposal
# goal:
# train 2 deep learning segmentation models:
# 1. passive learning model
# 2. active learning model
# then compare whether active learning actually helps or not
# active learning should use dropout-based model uncertainty
# evaluation should be image-level and include IoU / counts
# AWS is mainly for the final clean reproducible runs
# ============================================================


# ============================================================
# part 0. first clarify the new project direction
# ============================================================

# we are not making cellpose the main thing anymore.
# now the main question is:
# would a cell segmentation model be improved with active learning?
# so the core implementation is:
# - one passive segmentation model
# - one active segmentation model
# - same dataset
# - same architecture
# - same training budget
# - only difference is how new images get selected
# the meeting notes also say:
# - use batch active learning
# - use dropout for uncertainty
# - focus on model uncertainty, not augmentation disagreement
# - split data into train / validation / test
# - no cross-validation
# - UI is optional
# - AWS scripts/configs/final runs are expected later
# :contentReference[oaicite:0]{index=0} :contentReference[oaicite:1]{index=1}


# ============================================================
# part 1. dataset + preprocessing setup
# ============================================================

# step 1:
# load the U2OS nuclei dataset and make sure each image matches:
# - raw image
# - ground truth mask
# - cell count info if available
# the original proposal says the dataset includes U2OS nuclei images,
# z-stacks, focal plane info, masks, and cell count ground truth.
# :contentReference[oaicite:2]{index=2} :contentReference[oaicite:3]{index=3}

# step 2:
# decide if we are still using z-stack preprocessing or only in-focus images.
# if we keep the original preprocessing logic, then consistently select
# the middle z-plane / representative focal slice for each stack.
# original proposal had this as a preprocessing step.
# 

# step 3:
# normalize images so inputs are consistent.
# this means things like:
# - same image size
# - same number of channels
# - same intensity scaling / normalization
# - same datatype
# save preprocessing code cleanly because we will need the same thing
# for passive and active learning.
# 

# step 4:
# create a fixed split:
# - train pool
# - validation set
# - test set
# validation is needed for deep learning.
# test set stays untouched until evaluation.
# no cross-validation because that is too expensive.
# :contentReference[oaicite:6]{index=6}


# ============================================================
# part 2. passive learning implementation
# ============================================================

# this is the baseline.
# passive learning means image selection is random, not uncertainty-based.
# we need this so we have something fair to compare active learning against.
# both the original proposal and the 03/19 notes mention comparing to passive learning.
# 

# step 1:
# choose an initial labeled training subset from the train pool.
# do this randomly.
# save the selected image ids somewhere so the run is reproducible.

# step 2:
# build the segmentation training pipeline.
# this should include:
# - dataset class / dataloader
# - model
# - loss
# - optimizer
# - training loop
# - validation loop
# - checkpoint saving
# - test evaluation

# step 3:
# train the passive model on the initial labeled subset.
# save:
# - training loss
# - validation loss
# - checkpoint
# - predictions on validation set

# step 4:
# implement passive learning rounds.
# for each round:
# - randomly choose a new batch of unlabeled images
# - add them to the labeled training set
# - retrain or fine-tune the model
# - evaluate again
# batch size should match the active learning batch size
# so the comparison is fair.
# the meeting notes specifically say batch active learning,
# so passive should also grow in batches.
# :contentReference[oaicite:8]{index=8}

# step 5:
# evaluate passive model performance after each round.
# track at least:
# - IoU / Dice for segmentation
# - predicted count vs ground truth count
# - performance vs number of labeled images
# the meeting notes specifically mention IoU and counts.
# :contentReference[oaicite:9]{index=9}


# ============================================================
# part 3. active learning implementation
# ============================================================

# this is the actual experimental model.
# same architecture and same starting labeled set as passive.
# the only thing that changes is the image selection strategy.
# active learning should use dropout-based uncertainty.
# :contentReference[oaicite:10]{index=10}

# step 1:
# start with the exact same initial labeled set used in passive learning.
# do not change this or else the comparison gets messy.

# step 2:
# make sure the segmentation model supports dropout.
# we want dropout active during inference for uncertainty estimation.
# this lets us generate multiple stochastic predictions per image.
# meeting notes describe this as creating k perturbed models / predictions
# and checking variance across them.
# :contentReference[oaicite:11]{index=11}

# step 3:
# implement uncertainty estimation for one unlabeled image.
# for each candidate image:
# - run the model multiple times with dropout on
# - get multiple predicted masks
# - compare how much the predictions vary
# uncertainty should be measured at the image level, not cell level.
# possible image-level uncertainty signals:
# - variance in predicted masks
# - variance in predicted cell counts
# - disagreement across predictions
# the notes also say counts are useful for whole-image uncertainty.
# :contentReference[oaicite:12]{index=12}

# step 4:
# rank unlabeled images by uncertainty score.
# then choose the top-k most uncertain images as the next batch.
# this is the active learning selection rule.
# original proposal also described closed-loop batch selection.
# 

# step 5:
# implement the active learning loop.
# for each round:
# - train/fine-tune model on current labeled set
# - run dropout inference on unlabeled pool
# - compute image-level uncertainty for each candidate
# - rank candidates
# - select top-k uncertain images
# - add them to the labeled set
# - retrain / continue training
# - evaluate on validation and test sets

# step 6:
# save active learning metadata every round.
# save things like:
# - selected image ids
# - uncertainty scores
# - per-round metrics
# - checkpoints
# - plots / overlays if possible
# this will make report writing way easier later.


# ============================================================
# part 4. shared deep learning implementation
# ============================================================

# since the 03/19 update says the main new thing is that we are using
# deep learning to create the segmentation model, this part is really the backbone
# of both passive and active learning.
# :contentReference[oaicite:14]{index=14}

# step 1:
# choose a segmentation architecture we can realistically train.
# do not overcomplicate this.
# better to have a simple model that works cleanly than an ambitious one
# that never trains properly.

# step 2:
# define the loss function and evaluation metrics.
# metrics should include:
# - IoU or Dice
# - count comparison to ground truth
# because those are specifically mentioned in the meeting notes.
# :contentReference[oaicite:15]{index=15}

# step 3:
# create visualization outputs.
# we should save examples like:
# - raw image
# - ground truth mask
# - predicted mask
# - maybe uncertainty heat / summary if we can
# this will help with report figures and slides.

# step 4:
# make the code reproducible.
# use:
# - fixed random seeds
# - config files
# - saved split files
# - saved selected image lists
# - checkpoint naming by round / experiment name

# step 5:
# keep model code modular.
# ideal files might be something like:
# - dataset.py
# - model.py
# - train.py
# - evaluate.py
# - passive_loop.py
# - active_loop.py
# - uncertainty.py
# - utils.py
# so everything is not shoved into one messy notebook/script.


# ============================================================
# part 5. old proposal pieces that are now lower priority
# ============================================================

# these were part of the original proposal, but after the 03/19 meeting
# they are not the main thing anymore:
# - Cellpose as the core segmentation engine
# - augmentation-based uncertainty
# - image quality gate as a central module
# - morphology robustness module
# - UI
# these are optional / secondary now unless we have extra time.
# cellpose can still be used as a comparison baseline just for fun.
# UI is explicitly optional.
# 

# so priority order should be:
# 1. passive model works
# 2. active model works
# 3. uncertainty selection works
# 4. evaluation + comparison plots work
# 5. then optional extras if time permits


# ============================================================
# part 6. AWS implementation
# ============================================================

# AWS is mostly for scaling and final clean runs.
# original timeline says AWS scaling, AWS scripts/configs,
# closed-loop reproducible runs, and final export are expected.
# :contentReference[oaicite:17]{index=17}

# step 1:
# set up the AWS environment.
# make sure we have:
# - required packages installed
# - GPU access if needed
# - clean environment setup instructions
# - requirements.txt or environment file

# step 2:
# organize files cleanly in the cloud.
# suggested folders:
# - data/
# - splits/
# - configs/
# - checkpoints/
# - results/
# - plots/
# - logs/

# step 3:
# make config-driven scripts.
# important configs:
# - learning rate
# - batch size
# - epochs
# - number of active learning rounds
# - active learning batch size
# - number of dropout passes
# - random seed
# - output directory
# the timeline explicitly mentions AWS scripts + configs.
# :contentReference[oaicite:18]{index=18}

# step 4:
# run passive learning cleanly on AWS.
# save:
# - logs
# - metrics csv
# - checkpoints
# - plots

# step 5:
# run active learning cleanly on AWS.
# save:
# - selected image ids each round
# - uncertainty scores
# - metrics csv
# - checkpoints
# - plots

# step 6:
# generate final outputs for report/slides.
# the proposal timeline says final clean runs, finalized figures,
# reproducibility notes, parameter table, and export for slides/report.
# :contentReference[oaicite:19]{index=19}

# step 7:
# if feasible, run multiple seeds.
# this was mentioned as optional in the timeline for final runs.
# :contentReference[oaicite:20]{index=20}


# ============================================================
# part 7. best implementation order so we do not get lost
# ============================================================

# phase 1:
# get dataset loading + preprocessing working

# phase 2:
# make one segmentation model train successfully on a small subset

# phase 3:
# implement passive learning loop with random batch additions

# phase 4:
# implement dropout uncertainty code

# phase 5:
# implement active learning batch selection loop

# phase 6:
# compare passive vs active on a smaller test run first

# phase 7:
# move clean scripts to AWS

# phase 8:
# run final experiments and generate figures/tables


# ============================================================
# part 8. bare minimum deliverables we should end up with
# ============================================================

# passive learning:
# - random initial labeled set
# - random batch additions
# - segmentation training pipeline
# - per-round metrics

# active learning:
# - same initial labeled set
# - dropout uncertainty estimation
# - top-k uncertain image selection
# - per-round metrics

# deep learning:
# - segmentation model
# - train / val / test code
# - IoU / Dice / count evaluation
# - saved checkpoints and plots

# AWS:
# - reproducible scripts
# - configs
# - final runs
# - exported figures/tables for report


# ============================================================
# final note to ourselves
# ============================================================

# do not get distracted by optional polish too early.
# the actual grade-critical thing now is:
# can we show a clean comparison between passive learning and
# active learning for training a deep learning cell segmentation model?
# that is the heart of the modified proposal.
# 


