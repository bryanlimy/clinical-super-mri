## Data

### Data availability
We plan to host the dataset on a publicly accessible server and make it available upon request. In the meantime, please contact Dr. Maria del C. Valdés-Hernández ([M.Valdes-Hernan@ed.ac.uk](mailto:M.Valdes-Hernan@ed.ac.uk)) to inquire about the dataset used in this work.

### Data folder structure
- Data should have the following structure
    ```
    clinical-super-mri/
        data/
            affine/
                SR_002_NHSRI_V0_affine.mat
                SR_002_NHSRI_V1_affine.mat
                SR_005_BRIC1_V0_affine.mat
                ...
            rigid/
                SR_002_NHSRI_V0_rigid.mat
                SR_002_NHSRI_V1_rigid.mat
                SR_005_BRIC1_V0_rigid.mat
                ...
            warp/
                SR_002_NHSRI_V0.mat
                SR_002_NHSRI_V1.mat
                SR_005_BRIC1_V0.mat
                ...
    ```

### Convert `.mat` to `.npy`
- Here we provide a simple Python script `mat2npy.py` to convert `.mat` files to `.npy` as  loading `.npy` files is much faster in Python.
- The follow command convert all `.mat` scans in `affine/` to `.npy` files and store in `affine/npy`
    ```
    python mat2npy.py --input_dir affine --output_dir affine/npy
    ```
- Note that our data reader still support reading directly from `.mat` files though the training speed might be bottlenecked by reading `.mat` files. 