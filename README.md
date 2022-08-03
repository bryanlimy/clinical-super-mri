## Deep Attention Super-Resolution of Brain Magnetic Resonance Images Acquired Under Clinical Protocols

Code for the paper ["Deep Attention Super-Resolution of Brain Magnetic Resonance Images Acquired Under Clinical Protocols"](https://www.frontiersin.org/articles/10.3389/fncom.2022.887633/abstract).

Authors: Bryan M. Li, Leonardo V. Castorina, Maria Del Carmen Valdés Hernández, Una Clancy, Stewart John Wiseman, Eleni Sakka, Amos J. Storkey, Daniela Jaime Garcia, Yajun Cheng, Fergus Doubal, Michael T. Thrippleton, Michael Stringer and Joanna M Wardlaw.

![lr-sr-hr-sample-animation](misc/animation.gif)

Please use the following BibTeX entry:
```
@article{lideep,
  title={Deep Attention Super-Resolution of Brain Magnetic Resonance Images Acquired Under Clinical Protocols},
  author={Li, Bryan M and Castorina, Leonardo V and Vald{\'e}s Hern{\'a}ndez, Maria Del Carmen and Clancy, Una and Wiseman, Stewart John and Sakka, Eleni and Storkey, Amos J and Jaime Garcia, Daniela and Cheng, Yajun and Doubal, Fergus and others},
  journal={Frontiers in Computational Neuroscience},
  pages={97},
  publisher={Frontiers}
}
```

### Installation
- Create new [conda](https://docs.conda.io/en/latest/miniconda.html) environment `supermri`
    ```
    conda create -n supermri python=3.8
    ```
- Activate `supermri` environment
    ```
    conda activate supermri
    ```
- Run installation script
    ```
    sh setup.sh
    ```

### Dataset
- See [data/README.md](data/README.md) regarding data availability and the structure of the dataset.

### Train model
- [`train.py`](train.py) contains vast majority of the model training procedure, see the options below:
    ```
    usage: train.py [-h] --input_dir INPUT_DIR [--extension {npy,mat}] [--patch_size PATCH_SIZE] [--n_patches N_PATCHES] [--combine_sequence] [--model MODEL] [--num_filters NUM_FILTERS] [--normalization NORMALIZATION] [--activation ACTIVATION]
                    [--dropout DROPOUT] [--critic CRITIC] [--critic_num_filters CRITIC_NUM_FILTERS] [--critic_num_blocks CRITIC_NUM_BLOCKS] [--critic_dropout CRITIC_DROPOUT] [--critic_lr CRITIC_LR] [--critic_steps CRITIC_STEPS]
                    [--critic_intensity CRITIC_INTENSITY] [--label_smoothing] [--lr LR] [--lr_step_size LR_STEP_SIZE] [--lr_gamma LR_GAMMA] --output_dir OUTPUT_DIR [--batch_size N] [--epochs EPOCHS] [--loss LOSS] [--no_cuda] [--seed SEED]
                    [--mixed_precision] [--num_workers NUM_WORKERS] [--save_plots] [--dpi DPI] [--clear_output_dir] [--verbose {0,1,2}]
    
    optional arguments:
      -h, --help            show this help message and exit
      --input_dir INPUT_DIR
                            path to directory with .npy or .mat files
      --extension {npy,mat}
                            MRI scan file extension
      --patch_size PATCH_SIZE
                            patch size, None to train on the entire scan.
      --n_patches N_PATCHES
                            number of patches to generate per sample, None to use all patches.
      --combine_sequence    combine FLAIR, T1 and T2 as a single input
      --model MODEL         model to use
      --num_filters NUM_FILTERS
                            number of filters or hidden units (default: 64)
      --normalization NORMALIZATION
                            normalization layer (default: instancenorm)
      --activation ACTIVATION
                            activation layer (default: leakyrelu)
      --dropout DROPOUT     dropout rate (default 0.0)
      --critic CRITIC       adversarial loss to use.
      --critic_num_filters CRITIC_NUM_FILTERS
                            number of filters or hidden units in critic model
      --critic_num_blocks CRITIC_NUM_BLOCKS
                            number of blocks in DCGAN critic model
      --critic_dropout CRITIC_DROPOUT
                            critic model dropout rate
      --critic_lr CRITIC_LR
                            critic model learning rate
      --critic_steps CRITIC_STEPS
                            number of update steps for critic per global step
      --critic_intensity CRITIC_INTENSITY
                            critic score coefficient when training the up-sampling model.
      --label_smoothing     label smoothing in critic loss calculation
      --lr LR               learning rate (default: 0.001)
      --lr_step_size LR_STEP_SIZE
                            learning rate decay step size (default: 20)
      --lr_gamma LR_GAMMA   learning rate step gamma (default: 0.5)
      --output_dir OUTPUT_DIR
                            directory to write TensorBoard summary.
      --batch_size N        batch size for training (default: 32)
      --epochs EPOCHS       number of epochs (default: 100)
      --loss LOSS           loss function to use (default: bce)
      --no_cuda             disables CUDA training
      --seed SEED           random seed (default: 42)
      --mixed_precision     use mixed precision training
      --num_workers NUM_WORKERS
                            number of workers for data loader
      --save_plots          save TensorBoard figures and images to disk.
      --dpi DPI             DPI of matplotlib figures
      --clear_output_dir    overwrite output directory if exists
      --verbose {0,1,2}     verbosity. 0 - no print statement, 2 - print all print statements.
    ```
- The following is an example on how to train a `AGUNet` on `affine` data.
- Activate `supermri` conda environment
    ```
    conda activate supermri
    ```
- Assume we store all the `affine` scans in `dataset/affine` in `.mat` format. We train `AGUNet` up-sampling model with `DCGAN` critic model to guide the model training for 100 epochs, and store model checkpoint and summary to `runs/affine/001_agunet` with the following command
    ```
    python train.py --input_dir data/affine --output_dir runs/001_agunet_affine --model agunet --num_filters 16 --combine_sequence --critic dcgan --critic_num_filters 16 --critic_intensity 0.1 --batch_size 32 --epochs 100 --extension mat --mixed_precision
    ```

### Monitoring and Visualization
- To monitor training performance and up-sampled results stored in `--output_dir`, we can use the following command
    ```
    tensorboard --logdir runs/001_agunet_affine
    ```
- By default, TensorBoard starts a local server at port `6006`, you can check the TensorBoard summary by visiting `localhost:6006`.

### Inference
- After training the model with checkpoint saved, you can use `predict.py` to up-sample any scans in `.mat` format.
- In terminal, navigate to the repository home folder.
- Activate `conda` environment
    ```
    conda activate supermri
    ```
- To up-sample low-resolution `affine` scans using model checkpoint in `runs/001_agunet_affine`, we can use the following command:
    ```
    python predict.py --input_dir <path to directory with .mat files> --model_dir runs/001_agunet_affine --output_dir <path to store up-sample scans>
    ```
- The up-sampled scans would be stored under `--output_dir` with the same name as the original scan.
- Scans **must** have keys: `{FLAIRarray, T1array, T2array}` in the `.mat` file.
