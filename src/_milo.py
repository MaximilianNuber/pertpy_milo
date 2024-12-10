from __future__ import annotations

import random
import re
from typing import TYPE_CHECKING, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from anndata import AnnData
from lamin_utils import logger
from mudata import MuData
from formulaic import model_matrix

# ._doc import _doc_params, doc_common_plot_args

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes
    from matplotlib.colors import Colormap
    from matplotlib.figure import Figure

from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import euclidean_distances


class Milo:
    """Python implementation of Milo."""

    def __init__(self):
        try:
            from rpy2.robjects import conversion, numpy2ri, pandas2ri
            from rpy2.robjects.packages import STAP, PackageNotInstalledError, importr
        except ModuleNotFoundError:
            raise ImportError("milo requires rpy2 to be installed.") from None

        try:
            importr("edgeR")
        except ImportError as e:
            raise ImportError("milo requires a valid R installation with edger installed:\n") from e

    def load(
        self,
        input: AnnData,
        feature_key: str | None = "rna",
    ) -> MuData:
        """Prepare a MuData object for subsequent processing.

        Args:
            input: AnnData
            feature_key: Key to store the cell-level AnnData object in the MuData object
        Returns:
            MuData: MuData object with original AnnData.

        Examples:
            >>> import pertpy as pt
            >>> adata = pt.dt.bhattacherjee()
            >>> milo = pt.tl.Milo()
            >>> mdata = milo.load(adata)
        """
        print("entirely new 5")
        mdata = MuData({feature_key: input, "milo": AnnData()})

        return mdata

    def make_nhoods(
        self,
        data: AnnData | MuData,
        neighbors_key: str | None = None,
        feature_key: str | None = "rna",
        prop: float = 0.1,
        seed: int = 0,
        copy: bool = False,
    ):
        """Randomly sample vertices on a KNN graph to define neighbourhoods of cells.

        The set of neighborhoods get refined by computing the median profile for the neighbourhood in reduced dimensional space
        and by selecting the nearest vertex to this position.
        Thus, multiple neighbourhoods may be collapsed to prevent over-sampling the graph space.

        Args:
            data: AnnData object with KNN graph defined in `obsp` or MuData object with a modality with KNN graph defined in `obsp`
            neighbors_key: The key in `adata.obsp` or `mdata[feature_key].obsp` to use as KNN graph.
                           If not specified, `make_nhoods` looks .obsp[‘connectivities’] for connectivities (default storage places for `scanpy.pp.neighbors`).
                           If specified, it looks at .obsp[.uns[neighbors_key][‘connectivities_key’]] for connectivities.
            feature_key: If input data is MuData, specify key to cell-level AnnData object.
            prop: Fraction of cells to sample for neighbourhood index search.
            seed: Random seed for cell sampling.
            copy: Determines whether a copy of the `adata` is returned.

        Returns:
            If `copy=True`, returns the copy of `adata` with the result in `.obs`, `.obsm`, and `.uns`.
            Otherwise:

            nhoods: scipy.sparse._csr.csr_matrix in `adata.obsm['nhoods']`.
            A binary matrix of cell to neighbourhood assignments. Neighbourhoods in the columns are ordered by the order of the index cell in adata.obs_names

            nhood_ixs_refined: pandas.Series in `adata.obs['nhood_ixs_refined']`.
            A boolean indicating whether a cell is an index for a neighbourhood

            nhood_kth_distance: pandas.Series in `adata.obs['nhood_kth_distance']`.
            The distance to the kth nearest neighbour for each index cell (used for SpatialFDR correction)

            nhood_neighbors_key: `adata.uns["nhood_neighbors_key"]`
            KNN graph key, used for neighbourhood construction

        Examples:
            >>> import pertpy as pt
            >>> import scanpy as sc
            >>> adata = pt.dt.bhattacherjee()
            >>> milo = pt.tl.Milo()
            >>> mdata = milo.load(adata)
            >>> sc.pp.neighbors(mdata["rna"])
            >>> milo.make_nhoods(mdata["rna"])
        """
        if isinstance(data, MuData):
            adata = data[feature_key]
        if isinstance(data, AnnData):
            adata = data
        if copy:
            adata = adata.copy()

        # Get reduced dim used for KNN graph
        if neighbors_key is None:
            try:
                use_rep = adata.uns["neighbors"]["params"]["use_rep"]
            except KeyError:
                logger.warning("Using X_pca as default embedding")
                use_rep = "X_pca"
            try:
                knn_graph = adata.obsp["connectivities"].copy()
            except KeyError:
                logger.error('No "connectivities" slot in adata.obsp -- please run scanpy.pp.neighbors(adata) first')
                raise
        else:
            try:
                use_rep = adata.uns[neighbors_key]["params"]["use_rep"]
            except KeyError:
                logger.warning("Using X_pca as default embedding")
                use_rep = "X_pca"
            knn_graph = adata.obsp[neighbors_key + "_connectivities"].copy()

        X_dimred = adata.obsm[use_rep]
        n_ixs = int(np.round(adata.n_obs * prop))
        knn_graph[knn_graph != 0] = 1
        random.seed(seed)
        random_vertices = random.sample(range(adata.n_obs), k=n_ixs)
        random_vertices.sort()
        ixs_nn = knn_graph[random_vertices, :]
        non_zero_rows = ixs_nn.nonzero()[0]
        non_zero_cols = ixs_nn.nonzero()[1]
        refined_vertices = np.empty(
            shape=[
                len(random_vertices),
            ]
        )

        for i in range(len(random_vertices)):
            nh_pos = np.median(X_dimred[non_zero_cols[non_zero_rows == i], :], 0).reshape(-1, 1)
            nn_ixs = non_zero_cols[non_zero_rows == i]
            # Find closest real point (amongst nearest neighbors)
            dists = euclidean_distances(X_dimred[non_zero_cols[non_zero_rows == i], :], nh_pos.T)
            # Update vertex index
            refined_vertices[i] = nn_ixs[dists.argmin()]

        refined_vertices = np.unique(refined_vertices.astype("int"))
        refined_vertices.sort()

        nhoods = knn_graph[:, refined_vertices]
        adata.obsm["nhoods"] = nhoods

        # Add ixs to adata
        adata.obs["nhood_ixs_random"] = adata.obs_names.isin(adata.obs_names[random_vertices])
        adata.obs["nhood_ixs_refined"] = adata.obs_names.isin(adata.obs_names[refined_vertices])
        adata.obs["nhood_ixs_refined"] = adata.obs["nhood_ixs_refined"].astype("int")
        adata.obs["nhood_ixs_random"] = adata.obs["nhood_ixs_random"].astype("int")
        adata.uns["nhood_neighbors_key"] = neighbors_key
        # Store distance to K-th nearest neighbor (used for spatial FDR correction)
        if neighbors_key is None:
            knn_dists = adata.obsp["distances"]
        else:
            knn_dists = adata.obsp[neighbors_key + "_distances"]

        nhood_ixs = adata.obs["nhood_ixs_refined"] == 1
        dist_mat = knn_dists[nhood_ixs, :]
        k_distances = dist_mat.max(1).toarray().ravel()
        adata.obs["nhood_kth_distance"] = 0
        adata.obs["nhood_kth_distance"] = adata.obs["nhood_kth_distance"].astype(float)
        adata.obs.loc[adata.obs["nhood_ixs_refined"] == 1, "nhood_kth_distance"] = k_distances

        if copy:
            return adata

    def count_nhoods(
        self,
        data: AnnData | MuData,
        sample_col: str,
        feature_key: str | None = "rna",
    ):
        """Builds a sample-level AnnData object storing the matrix of cell counts per sample per neighbourhood.

        Args:
            data: AnnData object with neighbourhoods defined in `obsm['nhoods']` or MuData object with a modality with neighbourhoods defined in `obsm['nhoods']`
            sample_col: Column in adata.obs that contains sample information
            feature_key: If input data is MuData, specify key to cell-level AnnData object.

        Returns:
            MuData object storing the original (i.e. rna) AnnData in `mudata[feature_key]`
            and the compositional anndata storing the neighbourhood cell counts in `mudata['milo']`.
            Here:
            - `mudata['milo'].obs_names` are samples (defined from `adata.obs['sample_col']`)
            - `mudata['milo'].var_names` are neighbourhoods
            - `mudata['milo'].X` is the matrix counting the number of cells from each
            sample in each neighbourhood

        Examples:
            >>> import pertpy as pt
            >>> import scanpy as sc
            >>> adata = pt.dt.bhattacherjee()
            >>> milo = pt.tl.Milo()
            >>> mdata = milo.load(adata)
            >>> sc.pp.neighbors(mdata["rna"])
            >>> milo.make_nhoods(mdata["rna"])
            >>> mdata = milo.count_nhoods(mdata, sample_col="orig.ident")
        """
        if isinstance(data, MuData):
            adata = data[feature_key]
            is_MuData = True
        if isinstance(data, AnnData):
            adata = data
            is_MuData = False
        if isinstance(adata, AnnData):
            try:
                nhoods = adata.obsm["nhoods"]
            except KeyError:
                logger.error('Cannot find "nhoods" slot in adata.obsm -- please run milopy.make_nhoods(adata)')
                raise
        # Make nhood abundance matrix
        sample_dummies = pd.get_dummies(adata.obs[sample_col])
        all_samples = sample_dummies.columns
        sample_dummies = csr_matrix(sample_dummies.values)
        nhood_count_mat = nhoods.T.dot(sample_dummies)
        sample_obs = pd.DataFrame(index=all_samples)
        sample_adata = AnnData(X=nhood_count_mat.T, obs=sample_obs)
        sample_adata.uns["sample_col"] = sample_col
        # Save nhood index info
        sample_adata.var["index_cell"] = adata.obs_names[adata.obs["nhood_ixs_refined"] == 1]
        sample_adata.var["kth_distance"] = adata.obs.loc[
            adata.obs["nhood_ixs_refined"] == 1, "nhood_kth_distance"
        ].values

        if is_MuData is True:
            data.mod["milo"] = sample_adata
            return data
        else:
            milo_mdata = MuData({feature_key: adata, "milo": sample_adata})
            return milo_mdata

    def da_nhoods(
        self,
        mdata: MuData,
        design: str,
        model_contrasts: str | None = None,
        subset_samples: list[str] | None = None,
        add_intercept: bool = True,
        feature_key: str | None = "rna",
        solver: Literal["edger", "batchglm"] = "edger",
    ):
        """Performs differential abundance testing on neighbourhoods using QLF test implementation as implemented in edgeR.

        Args:
            mdata: MuData object
            design: Formula for the test, following glm syntax from R (e.g. '~ condition').
                    Terms should be columns in `milo_mdata[feature_key].obs`.
            model_contrasts: A string vector that defines the contrasts used to perform DA testing, following glm syntax from R (e.g. "conditionDisease - conditionControl").
                             If no contrast is specified (default), then the last categorical level in condition of interest is used as the test group.
            subset_samples: subset of samples (obs in `milo_mdata['milo']`) to use for the test.
            add_intercept: whether to include an intercept in the model. If False, this is equivalent to adding + 0 in the design formula. When model_contrasts is specified, this is set to False by default.
            feature_key: If input data is MuData, specify key to cell-level AnnData object.
            solver: The solver to fit the model to. One of "edger" (requires R, rpy2 and edgeR to be installed) or "batchglm"

        Returns:
            None, modifies `milo_mdata['milo']` in place, adding the results of the DA test to `.var`:
            - `logFC` stores the log fold change in cell abundance (coefficient from the GLM)
            - `PValue` stores the p-value for the QLF test before multiple testing correction
            - `SpatialFDR` stores the p-value adjusted for multiple testing to limit the false discovery rate,
                calculated with weighted Benjamini-Hochberg procedure

        Examples:
            >>> import pertpy as pt
            >>> import scanpy as sc
            >>> adata = pt.dt.bhattacherjee()
            >>> milo = pt.tl.Milo()
            >>> mdata = milo.load(adata)
            >>> sc.pp.neighbors(mdata["rna"])
            >>> milo.make_nhoods(mdata["rna"])
            >>> mdata = milo.count_nhoods(mdata, sample_col="orig.ident")
            >>> milo.da_nhoods(mdata, design="~label")
        """
        try:
            sample_adata = mdata["milo"]
        except KeyError:
            logger.error(
                "milo_mdata should be a MuData object with two slots:"
                " feature_key and 'milo' - please run milopy.count_nhoods() first"
            )
            raise
        adata = mdata[feature_key]

        covariates = [x.strip(" ") for x in set(re.split("\\+|\\*|\\:", design.lstrip("~ ")))]

        # Add covariates used for testing to sample_adata.var
        sample_col = sample_adata.uns["sample_col"]
        try:
            sample_obs = adata.obs[covariates + [sample_col]].drop_duplicates()
        except KeyError:
            missing_cov = [x for x in covariates if x not in sample_adata.obs.columns]
            logger.warning("Covariates {c} are not columns in adata.obs".format(c=" ".join(missing_cov)))
            raise
        sample_obs = sample_obs[covariates + [sample_col]]
        sample_obs.index = sample_obs[sample_col].astype("str")

        try:
            assert sample_obs.loc[sample_adata.obs_names].shape[0] == len(sample_adata.obs_names)
        except AssertionError:
            logger.warning(
                f"Values in mdata[{feature_key}].obs[{covariates}] cannot be unambiguously assigned to each sample"
                f" -- each sample value should match a single covariate value"
            )
            raise
        sample_adata.obs = sample_obs.loc[sample_adata.obs_names]

        # Get design dataframe
        try:
            design_df = sample_adata.obs[covariates]
        except KeyError:
            missing_cov = [x for x in covariates if x not in sample_adata.obs.columns]
            logger.error(
                'Covariates {c} are not columns in adata.uns["sample_adata"].obs'.format(c=" ".join(missing_cov))
            )
            raise
        # Get count matrix
        count_mat = sample_adata.X.T.toarray()
        print(count_mat.shape)
        lib_size = count_mat.sum(0)

        # Filter out samples with zero counts
        keep_smp = lib_size > 0

        # Subset samples
        if subset_samples is not None:
            keep_smp = keep_smp & sample_adata.obs_names.isin(subset_samples)
            design_df = design_df[keep_smp]
            for i, e in enumerate(design_df.columns):
                if design_df.dtypes[i].name == "category":
                    design_df[e] = design_df[e].cat.remove_unused_categories()

        # Filter out nhoods with zero counts (they can appear after sample filtering)
        keep_nhoods = count_mat[:, keep_smp].sum(1) > 0

        if solver == "edger":
            # Set up rpy2 to run edgeR
            edgeR, limma, stats, base = self._setup_rpy2()

            # Define model matrix
            if not add_intercept or model_contrasts is not None:
                design = design + " + 0"
            model = model_matrix(design, design_df)
            print(model.columns)

            model.columns = [''.join(list(re.split("\\[|\\.|\\]|T", x))) for x in model.columns]
            model.columns = [x.replace(":", ".") for x in model.columns]
            

            # Fit NB-GLM
            dge = edgeR.DGEList(
                counts=_py_to_r(count_mat[keep_nhoods, :][:, keep_smp]), 
                lib_size=_py_to_r(lib_size[keep_smp])
            )
            dge = edgeR.calcNormFactors(dge, method="TMM")

            r_model = _py_to_r(model)
            
            dge = edgeR.estimateDisp(dge, r_model)
            
            fit = edgeR.glmQLFit(dge, base.as_matrix(r_model), robust=_py_to_r(True))

            # Test
            n_coef = model.shape[1]
            if model_contrasts is not None:
                r_str = """
                get_model_cols <- function(design_df, design){
                    m = model.matrix(object=formula(design), data=design_df)
                    return(colnames(m))
                }
                """
                from rpy2.robjects.packages import STAP

                get_model_cols = STAP(r_str, "get_model_cols")
                model_mat_cols = list(model.columns)
                print(model_mat_cols)
                
                model_df = pd.DataFrame(model)
                
                model_df.columns = model_mat_cols

                r_model_df = _py_to_r(model_df)
                try:
                    
                    mod_contrast = limma.makeContrasts(contrasts=_py_to_r(model_contrasts), levels=r_model_df)
                except ValueError:
                    logger.error("Model contrasts must be in the form 'A-B' or 'A+B'")
                    raise
                res = base.as_data_frame(
                    edgeR.topTags(edgeR.glmQLFTest(fit, contrast=mod_contrast), sort_by="none", n=np.inf)
                )
            else:
                res = base.as_data_frame(edgeR.topTags(edgeR.glmQLFTest(fit, coef=n_coef), sort_by="none", n=np.inf))

            from rpy2.robjects import conversion

            res = _r_to_py(res)
            # if not isinstance(res, pd.DataFrame):
            #     res = pd.DataFrame(res)

        # Save outputs
        print(res.head(20))
        res.index = sample_adata.var_names[keep_nhoods]  # type: ignore
        
        if any(col in sample_adata.var.columns for col in res.columns):
            sample_adata.var = sample_adata.var.drop([x for x in res.columns if x in sample_adata.var.columns], axis=1)
        sample_adata.var = pd.concat([sample_adata.var, res], axis=1)

        # Run Graph spatial FDR correction
        self._graph_spatial_fdr(sample_adata, neighbors_key=adata.uns["nhood_neighbors_key"])

    def annotate_nhoods(
        self,
        mdata: MuData,
        anno_col: str,
        feature_key: str | None = "rna",
    ):
        """Assigns a categorical label to neighbourhoods, based on the most frequent label among cells in each neighbourhood. This can be useful to stratify DA testing results by cell types or samples.

        Args:
            mdata: MuData object
            anno_col: Column in adata.obs containing the cell annotations to use for nhood labelling
            feature_key: If input data is MuData, specify key to cell-level AnnData object.

        Returns:
            None. Adds in place:
            - `milo_mdata['milo'].var["nhood_annotation"]`: assigning a label to each nhood
            - `milo_mdata['milo'].var["nhood_annotation_frac"]` stores the fraciton of cells in the neighbourhood with the assigned label
            - `milo_mdata['milo'].varm['frac_annotation']`: stores the fraction of cells from each label in each nhood
            - `milo_mdata['milo'].uns["annotation_labels"]`: stores the column names for `milo_mdata['milo'].varm['frac_annotation']`

        Examples:
            >>> import pertpy as pt
            >>> import scanpy as sc
            >>> adata = pt.dt.bhattacherjee()
            >>> milo = pt.tl.Milo()
            >>> mdata = milo.load(adata)
            >>> sc.pp.neighbors(mdata["rna"])
            >>> milo.make_nhoods(mdata["rna"])
            >>> mdata = milo.count_nhoods(mdata, sample_col="orig.ident")
            >>> milo.annotate_nhoods(mdata, anno_col="cell_type")
        """
        try:
            sample_adata = mdata["milo"]
        except KeyError:
            logger.error(
                "milo_mdata should be a MuData object with two slots: feature_key and 'milo' - please run milopy.count_nhoods(adata) first"
            )
            raise
        adata = mdata[feature_key]

        # Check value is not numeric
        if pd.api.types.is_numeric_dtype(adata.obs[anno_col]):
            raise ValueError(
                "adata.obs[anno_col] is not of categorical type - please use milopy.utils.annotate_nhoods_continuous for continuous variables"
            )

        anno_dummies = pd.get_dummies(adata.obs[anno_col])
        anno_count = adata.obsm["nhoods"].T.dot(csr_matrix(anno_dummies.values))
        anno_count_dense = anno_count.toarray()
        anno_sum = anno_count_dense.sum(1)
        anno_frac = np.divide(anno_count_dense, anno_sum[:, np.newaxis])

        anno_frac_dataframe = pd.DataFrame(anno_frac, columns=anno_dummies.columns, index=sample_adata.var_names)
        sample_adata.varm["frac_annotation"] = anno_frac_dataframe.values
        sample_adata.uns["annotation_labels"] = anno_frac_dataframe.columns
        sample_adata.uns["annotation_obs"] = anno_col
        sample_adata.var["nhood_annotation"] = anno_frac_dataframe.idxmax(1)
        sample_adata.var["nhood_annotation_frac"] = anno_frac_dataframe.max(1)

    def annotate_nhoods_continuous(self, mdata: MuData, anno_col: str, feature_key: str | None = "rna"):
        """Assigns a continuous value to neighbourhoods, based on mean cell level covariate stored in adata.obs. This can be useful to correlate DA log-foldChanges with continuous covariates such as pseudotime, gene expression scores etc...

        Args:
            mdata: MuData object
            anno_col: Column in adata.obs containing the cell annotations to use for nhood labelling
            feature_key: If input data is MuData, specify key to cell-level AnnData object.

        Returns:
            None. Adds in place:
            - `milo_mdata['milo'].var["nhood_{anno_col}"]`: assigning a continuous value to each nhood

        Examples:
            >>> import pertpy as pt
            >>> import scanpy as sc
            >>> adata = pt.dt.bhattacherjee()
            >>> milo = pt.tl.Milo()
            >>> mdata = milo.load(adata)
            >>> sc.pp.neighbors(mdata["rna"])
            >>> milo.make_nhoods(mdata["rna"])
            >>> mdata = milo.count_nhoods(mdata, sample_col="orig.ident")
            >>> milo.annotate_nhoods_continuous(mdata, anno_col="nUMI")
        """
        if "milo" not in mdata.mod:
            raise ValueError(
                "milo_mdata should be a MuData object with two slots: feature_key and 'milo' - please run milopy.count_nhoods(adata) first"
            )
        adata = mdata[feature_key]

        # Check value is not categorical
        if not pd.api.types.is_numeric_dtype(adata.obs[anno_col]):
            raise ValueError(
                "adata.obs[anno_col] is not of continuous type - please use milopy.utils.annotate_nhoods for categorical variables"
            )

        anno_val = adata.obsm["nhoods"].T.dot(csr_matrix(adata.obs[anno_col]).T)

        mean_anno_val = anno_val.toarray() / np.array(adata.obsm["nhoods"].T.sum(1))

        mdata["milo"].var[f"nhood_{anno_col}"] = mean_anno_val

    def add_covariate_to_nhoods_var(self, mdata: MuData, new_covariates: list[str], feature_key: str | None = "rna"):
        """Add covariate from cell-level obs to sample-level obs. These should be covariates for which a single value can be assigned to each sample.

        Args:
            mdata: MuData object
            new_covariates: columns in `milo_mdata[feature_key].obs` to add to `milo_mdata['milo'].obs`.
            feature_key: If input data is MuData, specify key to cell-level AnnData object.

        Returns:
            None, adds columns to `milo_mdata['milo']` in place

        Examples:
            >>> import pertpy as pt
            >>> import scanpy as sc
            >>> adata = pt.dt.bhattacherjee()
            >>> milo = pt.tl.Milo()
            >>> mdata = milo.load(adata)
            >>> sc.pp.neighbors(mdata["rna"])
            >>> milo.make_nhoods(mdata["rna"])
            >>> mdata = milo.count_nhoods(mdata, sample_col="orig.ident")
            >>> milo.add_covariate_to_nhoods_var(mdata, new_covariates=["label"])
        """
        try:
            sample_adata = mdata["milo"]
        except KeyError:
            logger.error(
                "milo_mdata should be a MuData object with two slots: feature_key and 'milo' - please run milopy.count_nhoods(adata) first"
            )
            raise
        adata = mdata[feature_key]

        sample_col = sample_adata.uns["sample_col"]
        covariates = list(
            set(sample_adata.obs.columns[sample_adata.obs.columns != sample_col].tolist() + new_covariates)
        )
        try:
            sample_obs = adata.obs[covariates + [sample_col]].drop_duplicates()
        except KeyError:
            missing_cov = [covar for covar in covariates if covar not in sample_adata.obs.columns]
            logger.error("Covariates {c} are not columns in adata.obs".format(c=" ".join(missing_cov)))
            raise
        sample_obs = sample_obs[covariates + [sample_col]].astype("str")
        sample_obs.index = sample_obs[sample_col]
        try:
            assert sample_obs.loc[sample_adata.obs_names].shape[0] == len(sample_adata.obs_names)
        except ValueError:
            logger.error(
                "Covariates cannot be unambiguously assigned to each sample -- each sample value should match a single covariate value"
            )
            raise
        sample_adata.obs = sample_obs.loc[sample_adata.obs_names]

    def build_nhood_graph(self, mdata: MuData, basis: str = "X_umap", feature_key: str | None = "rna"):
        """Build graph of neighbourhoods used for visualization of DA results

        Args:
            mdata: MuData object
            basis: Name of the obsm basis to use for layout of neighbourhoods (key in `adata.obsm`).
            feature_key: If input data is MuData, specify key to cell-level AnnData object.

        Returns:
            - `milo_mdata['milo'].varp['nhood_connectivities']`: graph of overlap between neighbourhoods (i.e. no of shared cells)
            - `milo_mdata['milo'].var["Nhood_size"]`: number of cells in neighbourhoods

        Examples:
            >>> import pertpy as pt
            >>> import scanpy as sc
            >>> adata = pt.dt.bhattacherjee()
            >>> milo = pt.tl.Milo()
            >>> mdata = milo.load(adata)
            >>> sc.pp.neighbors(mdata["rna"])
            >>> sc.tl.umap(mdata["rna"])
            >>> milo.make_nhoods(mdata["rna"])
            >>> mdata = milo.count_nhoods(mdata, sample_col="orig.ident")
            >>> milo.build_nhood_graph(mdata)
        """
        adata = mdata[feature_key]
        # # Add embedding positions
        mdata["milo"].varm["X_milo_graph"] = adata[adata.obs["nhood_ixs_refined"] == 1].obsm[basis]
        # Add nhood size
        mdata["milo"].var["Nhood_size"] = np.array(adata.obsm["nhoods"].sum(0)).flatten()
        # Add adjacency graph
        mdata["milo"].varp["nhood_connectivities"] = adata.obsm["nhoods"].T.dot(adata.obsm["nhoods"])
        mdata["milo"].varp["nhood_connectivities"].setdiag(0)
        mdata["milo"].varp["nhood_connectivities"].eliminate_zeros()
        mdata["milo"].uns["nhood"] = {
            "connectivities_key": "nhood_connectivities",
            "distances_key": "",
        }

    def add_nhood_expression(self, mdata: MuData, layer: str | None = None, feature_key: str | None = "rna") -> None:
        """Calculates the mean expression in neighbourhoods of each feature.

        Args:
            mdata: MuData object
            layer: If provided, use `milo_mdata[feature_key][layer]` as expression matrix instead of `milo_mdata[feature_key].X`.
            feature_key: If input data is MuData, specify key to cell-level AnnData object.

        Returns:
            Updates adata in place to store the matrix of average expression in each neighbourhood in `milo_mdata['milo'].varm['expr']`

        Examples:
            >>> import pertpy as pt
            >>> import scanpy as sc
            >>> adata = pt.dt.bhattacherjee()
            >>> milo = pt.tl.Milo()
            >>> mdata = milo.load(adata)
            >>> sc.pp.neighbors(mdata["rna"])
            >>> milo.make_nhoods(mdata["rna"])
            >>> mdata = milo.count_nhoods(mdata, sample_col="orig.ident")
            >>> milo.add_nhood_expression(mdata)
        """
        try:
            sample_adata = mdata["milo"]
        except KeyError:
            logger.error(
                "milo_mdata should be a MuData object with two slots:"
                " feature_key and 'milo' - please run milopy.count_nhoods(adata) first"
            )
            raise
        adata = mdata[feature_key]

        # Get gene expression matrix
        if layer is None:
            X = adata.X
            expr_id = "expr"
        else:
            X = adata.layers[layer]
            expr_id = "expr_" + layer

        # Aggregate over nhoods -- taking the mean
        nhoods_X = X.T.dot(adata.obsm["nhoods"])
        nhoods_X = csr_matrix(nhoods_X / adata.obsm["nhoods"].toarray().sum(0))
        sample_adata.varm[expr_id] = nhoods_X.T

    def find_nhood_groups(
        self,
        mdata, 
        SpatialFDR_threshold = 0.1,
        merge_discord = False,
        max_lfc_delta=None,
        overlap=1,
        subset_nhoods: None | str = None,
    ):
        """Get igraph graph from adjacency matrix. Adjusted from scanpy louvain clustering."""
        nhs = mdata["rna"].obsm["nhoods"].copy()
        nhood_adj = mdata["milo"].varp["nhood_connectivities"].copy()
        da_res = mdata["milo"].var.copy()
        is_da = np.asarray(da_res.SpatialFDR <= SpatialFDR_threshold)

        if subset_nhoods is not None:
            print(da_res.shape)
            mask_da_res = (
                da_res
                .query(subset_nhoods)
                .copy()
            )
            mask = np.asarray([True if x in mask_da_res.index else False for x in da_res.index])
            da_res = da_res.loc[mask, :].copy()
            print(da_res.shape)
            # mask = da_res.index.values.astype(bool)
            print(mask.shape)
            nhs = nhs[:, mask].copy()
            print(mask.shape)
            print(nhood_adj.shape)
            nhood_adj = nhood_adj[:, mask]
            nhood_adj = nhood_adj[mask, :].copy()
            print(nhood_adj.shape)
            # is_da = np.asarray(da_res.SpatialFDR <= SpatialFDR_threshold)
            is_da = is_da[mask].copy()
            # print(is_da.shape)

        # da_res.loc[is_da, 'logFC'].values @ (da_res.loc[is_da, 'logFC']).T.values
        # print(da_res.loc[is_da, 'logFC'])

        if merge_discord is False:
            # discord_sign = np.sign(da_res.loc[is_da, 'logFC'].values @ (da_res.loc[is_da, 'logFC'])) < 0
            # Assuming da.res is a pandas DataFrame and is.da is a boolean array
            discord_sign = np.sign(da_res.loc[is_da, 'logFC'].values @ da_res.loc[is_da, 'logFC'].values.T) < 0
            # print(discord_sign.shape)
            # nhood_adj[is_da, is_da][discord_sign] <- 0
            # print(nhood_adj.shape)
            nhood_adj[(is_da, is_da)][discord_sign] = 0

        if overlap > 1:
            nhood_adj[nhood_adj < overlap] = 0

        if max_lfc_delta is not None:  
            lfc_diff = np.array([da_res['logFC'].values - x for x in da_res['logFC'].values])
            nhood_adj[np.abs(lfc_diff) > max_lfc_delta] = 0

        # binarise
        # nhood_adj = np.asarray((nhood_adj > 0) + 0)
        nhood_adj = (nhood_adj > 0).astype(int)

        g = sc._utils.get_igraph_from_adjacency(nhood_adj, directed = False)

        weights = None
        part = g.community_multilevel(weights=weights)

        groups = np.array(part.membership)
        groups = groups.astype(int).astype(str)

        if subset_nhoods is not None:
            mdata["milo"].var["nhood_groups"] = np.nan
            mdata["milo"].var.loc[mask, "nhood_groups"] = groups
            return None
        mdata["milo"].var["nhood_groups"] = groups

    def _setup_rpy2(
        self,
    ):
        """Set up rpy2 to run edgeR"""
        from rpy2.robjects import numpy2ri, pandas2ri
        from rpy2.robjects.packages import importr

        edgeR = self._try_import_bioc_library("edgeR")
        limma = self._try_import_bioc_library("limma")
        stats = importr("stats")
        base = importr("base")

        return edgeR, limma, stats, base

    def _try_import_bioc_library(
        self,
        name: str,
    ):
        """Import R packages.

        Args:
            name (str): R packages name
        """
        from rpy2.robjects.packages import PackageNotInstalledError, importr

        try:
            _r_lib = importr(name)
            return _r_lib
        except PackageNotInstalledError:
            logger.error(f"Install Bioconductor library `{name!r}` first as `BiocManager::install({name!r}).`")
            raise

    def _graph_spatial_fdr(
        self,
        sample_adata: AnnData,
        neighbors_key: str | None = None,
    ):
        """FDR correction weighted on inverse of connectivity of neighbourhoods.

        The distance to the k-th nearest neighbor is used as a measure of connectivity.

        Args:
            sample_adata: Sample-level AnnData.
            neighbors_key: The key in `adata.obsp` to use as KNN graph.
        """
        # use 1/connectivity as the weighting for the weighted BH adjustment from Cydar
        w = 1 / sample_adata.var["kth_distance"]
        w[np.isinf(w)] = 0

        # Computing a density-weighted q-value.
        pvalues = sample_adata.var["PValue"]
        keep_nhoods = ~pvalues.isna()  # Filtering in case of test on subset of nhoods
        o = pvalues[keep_nhoods].argsort()
        pvalues = pvalues[keep_nhoods][o]
        w = w[keep_nhoods][o]

        adjp = np.zeros(shape=len(o))
        adjp[o] = (sum(w) * pvalues / np.cumsum(w))[::-1].cummin()[::-1]
        adjp = np.array([x if x < 1 else 1 for x in adjp])

        sample_adata.var["SpatialFDR"] = np.nan
        sample_adata.var.loc[keep_nhoods, "SpatialFDR"] = adjp

    # @_doc_params(common_plot_args=doc_common_plot_args)
    def plot_nhood_graph(
        self,
        mdata: MuData,
        *,
        alpha: float = 0.1,
        min_logFC: float = 0,
        min_size: int = 10,
        plot_edges: bool = False,
        title: str = "DA log-Fold Change",
        color_map: Colormap | str | None = None,
        palette: str | Sequence[str] | None = None,
        ax: Axes | None = None,
        show: bool = True,
        return_fig: bool = False,
        **kwargs,
    ) -> Figure | None:
        """Visualize DA results on abstracted graph (wrapper around sc.pl.embedding)

        Args:
            mdata: MuData object
            alpha: Significance threshold. (default: 0.1)
            min_logFC: Minimum absolute log-Fold Change to show results. If is 0, show all significant neighbourhoods.
            min_size: Minimum size of nodes in visualization. (default: 10)
            plot_edges: If edges for neighbourhood overlaps whould be plotted.
            title: Plot title.
            {common_plot_args}
            **kwargs: Additional arguments to `scanpy.pl.embedding`.

        Examples:
            >>> import pertpy as pt
            >>> import scanpy as sc
            >>> adata = pt.dt.bhattacherjee()
            >>> milo = pt.tl.Milo()
            >>> mdata = milo.load(adata)
            >>> sc.pp.neighbors(mdata["rna"])
            >>> sc.tl.umap(mdata["rna"])
            >>> milo.make_nhoods(mdata["rna"])
            >>> mdata = milo.count_nhoods(mdata, sample_col="orig.ident")
            >>> milo.da_nhoods(mdata,
            >>>            design='~label',
            >>>            model_contrasts='labelwithdraw_15d_Cocaine-labelwithdraw_48h_Cocaine')
            >>> milo.build_nhood_graph(mdata)
            >>> milo.plot_nhood_graph(mdata)

        Preview:
            .. image:: /_static/docstring_previews/milo_nhood_graph.png
        """
        nhood_adata = mdata["milo"].T.copy()

        if "Nhood_size" not in nhood_adata.obs.columns:
            raise KeyError(
                'Cannot find "Nhood_size" column in adata.uns["nhood_adata"].obs -- \
                    please run milopy.utils.build_nhood_graph(adata)'
            )

        nhood_adata.obs["graph_color"] = nhood_adata.obs["logFC"]
        nhood_adata.obs.loc[nhood_adata.obs["SpatialFDR"] > alpha, "graph_color"] = np.nan
        nhood_adata.obs["abs_logFC"] = abs(nhood_adata.obs["logFC"])
        nhood_adata.obs.loc[nhood_adata.obs["abs_logFC"] < min_logFC, "graph_color"] = np.nan

        # Plotting order - extreme logFC on top
        nhood_adata.obs.loc[nhood_adata.obs["graph_color"].isna(), "abs_logFC"] = np.nan
        ordered = nhood_adata.obs.sort_values("abs_logFC", na_position="first").index
        nhood_adata = nhood_adata[ordered]

        vmax = np.max([nhood_adata.obs["graph_color"].max(), abs(nhood_adata.obs["graph_color"].min())])
        vmin = -vmax

        fig = sc.pl.embedding(
            nhood_adata,
            "X_milo_graph",
            color="graph_color",
            cmap="RdBu_r",
            size=nhood_adata.obs["Nhood_size"] * min_size,
            edges=plot_edges,
            neighbors_key="nhood",
            sort_order=False,
            frameon=False,
            vmax=vmax,
            vmin=vmin,
            title=title,
            color_map=color_map,
            palette=palette,
            ax=ax,
            show=False,
            **kwargs,
        )

        if show:
            plt.show()
        if return_fig:
            return fig
        return None

    # @_doc_params(common_plot_args=doc_common_plot_args)
    def plot_nhood(
        self,
        mdata: MuData,
        ix: int,
        *,
        feature_key: str | None = "rna",
        basis: str = "X_umap",
        color_map: Colormap | str | None = None,
        palette: str | Sequence[str] | None = None,
        ax: Axes | None = None,
        show: bool = True,
        return_fig: bool = False,
        **kwargs,
    ) -> Figure | None:
        """Visualize cells in a neighbourhood.

        Args:
            mdata: MuData object with feature_key slot, storing neighbourhood assignments in `mdata[feature_key].obsm['nhoods']`
            ix: index of neighbourhood to visualize
            feature_key: Key in mdata to the cell-level AnnData object.
            basis: Embedding to use for visualization.
            color_map: Colormap to use for coloring.
            palette: Color palette to use for coloring.
            ax: Axes to plot on.
            {common_plot_args}
            **kwargs: Additional arguments to `scanpy.pl.embedding`.

        Examples:
            >>> import pertpy as pt
            >>> import scanpy as sc
            >>> adata = pt.dt.bhattacherjee()
            >>> milo = pt.tl.Milo()
            >>> mdata = milo.load(adata)
            >>> sc.pp.neighbors(mdata["rna"])
            >>> sc.tl.umap(mdata["rna"])
            >>> milo.make_nhoods(mdata["rna"])
            >>> milo.plot_nhood(mdata, ix=0)

        Preview:
            .. image:: /_static/docstring_previews/milo_nhood.png
        """
        mdata[feature_key].obs["Nhood"] = mdata[feature_key].obsm["nhoods"][:, ix].toarray().ravel()
        fig = sc.pl.embedding(
            mdata[feature_key],
            basis,
            color="Nhood",
            size=30,
            title="Nhood" + str(ix),
            color_map=color_map,
            palette=palette,
            return_fig=return_fig,
            ax=ax,
            show=False,
            **kwargs,
        )

        if show:
            plt.show()
        if return_fig:
            return fig
        return None

    # @_doc_params(common_plot_args=doc_common_plot_args)
    def plot_da_beeswarm(
        self,
        mdata: MuData,
        *,
        feature_key: str | None = "rna",
        anno_col: str = "nhood_annotation",
        alpha: float = 0.1,
        subset_nhoods: list[str] = None,
        palette: str | Sequence[str] | dict[str, str] | None = None,
        show: bool = True,
        return_fig: bool = False,
    ) -> Figure | None:
        """Plot beeswarm plot of logFC against nhood labels

        Args:
            mdata: MuData object
            feature_key: Key in mdata to the cell-level AnnData object.
            anno_col: Column in adata.uns['nhood_adata'].obs to use as annotation. (default: 'nhood_annotation'.)
            alpha: Significance threshold. (default: 0.1)
            subset_nhoods: List of nhoods to plot. If None, plot all nhoods.
            palette: Name of Seaborn color palette for violinplots.
                     Defaults to pre-defined category colors for violinplots.
            {common_plot_args}

        Returns:
            If `return_fig` is `True`, returns the figure, otherwise `None`.

        Examples:
            >>> import pertpy as pt
            >>> import scanpy as sc
            >>> adata = pt.dt.bhattacherjee()
            >>> milo = pt.tl.Milo()
            >>> mdata = milo.load(adata)
            >>> sc.pp.neighbors(mdata["rna"])
            >>> milo.make_nhoods(mdata["rna"])
            >>> mdata = milo.count_nhoods(mdata, sample_col="orig.ident")
            >>> milo.da_nhoods(mdata, design="~label")
            >>> milo.annotate_nhoods(mdata, anno_col="cell_type")
            >>> milo.plot_da_beeswarm(mdata)

        Preview:
            .. image:: /_static/docstring_previews/milo_da_beeswarm.png
        """
        try:
            nhood_adata = mdata["milo"].T.copy()
        except KeyError:
            raise RuntimeError(
                "mdata should be a MuData object with two slots: feature_key and 'milo'. Run 'milopy.count_nhoods(adata)' first."
            ) from None

        try:
            nhood_adata.obs[anno_col]
        except KeyError:
            raise RuntimeError(
                f"Unable to find {anno_col} in mdata['milo'].var. Run 'milopy.utils.annotate_nhoods(adata, anno_col)' first"
            ) from None

        if subset_nhoods is not None:
            nhood_adata = nhood_adata[nhood_adata.obs[anno_col].isin(subset_nhoods)]

        try:
            nhood_adata.obs["logFC"]
        except KeyError:
            raise RuntimeError(
                "Unable to find 'logFC' in mdata.uns['nhood_adata'].obs. Run 'core.da_nhoods(adata)' first."
            ) from None

        sorted_annos = (
            nhood_adata.obs[[anno_col, "logFC"]].groupby(anno_col).median().sort_values("logFC", ascending=True).index
        )

        anno_df = nhood_adata.obs[[anno_col, "logFC", "SpatialFDR"]].copy()
        anno_df["is_signif"] = anno_df["SpatialFDR"] < alpha
        anno_df = anno_df[anno_df[anno_col] != "nan"]

        try:
            obs_col = nhood_adata.uns["annotation_obs"]
            if palette is None:
                palette = dict(
                    zip(
                        mdata[feature_key].obs[obs_col].cat.categories,
                        mdata[feature_key].uns[f"{obs_col}_colors"],
                        strict=False,
                    )
                )
            sns.violinplot(
                data=anno_df,
                y=anno_col,
                x="logFC",
                order=sorted_annos,
                inner=None,
                orient="h",
                palette=palette,
                linewidth=0,
                scale="width",
            )
        except BaseException:  # noqa: BLE001
            sns.violinplot(
                data=anno_df,
                y=anno_col,
                x="logFC",
                order=sorted_annos,
                inner=None,
                orient="h",
                linewidth=0,
                scale="width",
            )
        sns.stripplot(
            data=anno_df,
            y=anno_col,
            x="logFC",
            order=sorted_annos,
            size=2,
            hue="is_signif",
            palette=["grey", "black"],
            orient="h",
            alpha=0.5,
        )
        plt.legend(loc="upper left", title=f"< {int(alpha * 100)}% SpatialFDR", bbox_to_anchor=(1, 1), frameon=False)
        plt.axvline(x=0, ymin=0, ymax=1, color="black", linestyle="--")

        if show:
            plt.show()
        if return_fig:
            return plt.gcf()
        return None

    # @_doc_params(common_plot_args=doc_common_plot_args)
    def plot_nhood_counts_by_cond(
        self,
        mdata: MuData,
        test_var: str,
        *,
        subset_nhoods: list[str] = None,
        log_counts: bool = False,
        show: bool = True,
        return_fig: bool = False,
    ) -> Figure | None:
        """Plot boxplot of cell numbers vs condition of interest.

        Args:
            mdata: MuData object storing cell level and nhood level information
            test_var: Name of column in adata.obs storing condition of interest (y-axis for boxplot)
            subset_nhoods: List of obs_names for neighbourhoods to include in plot. If None, plot all nhoods.
            log_counts: Whether to plot log1p of cell counts.
            {common_plot_args}

        Returns:
            If `return_fig` is `True`, returns the figure, otherwise `None`.
        """
        try:
            nhood_adata = mdata["milo"].T.copy()
        except KeyError:
            raise RuntimeError(
                "mdata should be a MuData object with two slots: feature_key and 'milo'. Run milopy.count_nhoods(mdata) first"
            ) from None

        if subset_nhoods is None:
            subset_nhoods = nhood_adata.obs_names

        pl_df = pd.DataFrame(nhood_adata[subset_nhoods].X.toarray(), columns=nhood_adata.var_names).melt(
            var_name=nhood_adata.uns["sample_col"], value_name="n_cells"
        )
        pl_df = pd.merge(pl_df, nhood_adata.var)
        pl_df["log_n_cells"] = np.log1p(pl_df["n_cells"])
        if not log_counts:
            sns.boxplot(data=pl_df, x=test_var, y="n_cells", color="lightblue")
            sns.stripplot(data=pl_df, x=test_var, y="n_cells", color="black", s=3)
            plt.ylabel("# cells")
        else:
            sns.boxplot(data=pl_df, x=test_var, y="log_n_cells", color="lightblue")
            sns.stripplot(data=pl_df, x=test_var, y="log_n_cells", color="black", s=3)
            plt.ylabel("log(# cells + 1)")

        plt.xticks(rotation=90)
        plt.xlabel(test_var)

        if show:
            plt.show()
        if return_fig:
            return plt.gcf()
        return None
    

#### Conversion functions

from scipy.sparse import csr_matrix
from anndata2ri import scipy2ri

from functools import singledispatch
from scipy.sparse import csc_matrix

import rpy2.rinterface_lib.callbacks
# import anndata2ri
import logging

from rpy2.robjects import pandas2ri
from rpy2.robjects import numpy2ri
from rpy2.robjects import r
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import default_converter
from rpy2.robjects.conversion import localconverter

from rpy2.robjects.packages import STAP

@singledispatch
def _py_to_r(object):
    with localconverter(default_converter):
        r_obj = numpy2ri.py2rpy(object)

    return r_obj

@_py_to_r.register
def _(object: np.ndarray):
    with localconverter(default_converter + numpy2ri.converter):
        r_obj = numpy2ri.py2rpy(object)
        print(type(r_obj))

    return r_obj

@_py_to_r.register
def _(object: pd.DataFrame):
    with localconverter(default_converter + pandas2ri.converter):
        r_obj = pandas2ri.py2rpy(object)

    return r_obj

@_py_to_r.register
def _(object: pd.DataFrame):
    with localconverter(default_converter + pandas2ri.converter):
        r_obj = pandas2ri.py2rpy(object)

    return r_obj

@_py_to_r.register
def _(object: csr_matrix):
    with localconverter(scipy2ri.converter):
        r_obj = scipy2ri.py2rpy(object)

    return r_obj

@_py_to_r.register
def _(object: csc_matrix):
    with localconverter(scipy2ri.converter):
        r_obj = scipy2ri.py2rpy(object)

    return r_obj

def _r_to_py(object):

    r_string = '''
    .get_class <- function(object) class(object)
    '''
    r_pkg = STAP(r_string, "r_pkg")
    
    cur_class = r_pkg._get_class(object)[0]
    
    if cur_class == "dgCMatrix":
        with localconverter(scipy2ri.converter):
            r_obj = scipy2ri.rpy2py(object)

        return r_obj

    if cur_class == "dgRMatrix":
        with localconverter(scipy2ri.converter):
            r_obj = scipy2ri.rpy2py(object)

        return r_obj

    if cur_class == "matrix":
        with localconverter(default_converter + numpy2ri.converter):
            r_obj = numpy2ri.rpy2py(object)
            
        return r_obj 

    if cur_class == "data.frame":
        with localconverter(default_converter + pandas2ri.converter):
            r_obj = pandas2ri.rpy2py(object)
            
        return r_obj 


# rbase = importr("base")

r_string = '''
.get_class <- function(object) class(object)
.get_colnames <- function(mat) colnames(mat)
.get_rownames <- function(mat) rownames(mat)
.get_rownames_dge <- function(dge) rownames(dge$counts)
'''
r_pkg = STAP(r_string, "r_pkg")

# edger = importr("edgeR")

def _ad_to_rmat(adata, layer = "X"):

    mat = adata.X if layer == "X" else adata.layers[layer]
    
    obsnames = np.asarray(adata.obs_names)
    obsnames = _py_to_r(obsnames)

    varnames = np.asarray(adata.var_names)
    varnames = _py_to_r(varnames)

    r_string = '''
    assign_colnames <- function(mat, names) {
        colnames(mat) <- names
        return(mat)
    }
    assign_rownames <- function(mat, names) {
        rownames(mat) <- names
        return(mat)
    }
    print_dimnames <- function(mat) {
        print(head(rownames(mat)))
    }

    .get_class <- function(object) class(object)
    .get_colnames <- function(mat) colnames(mat)
    .get_rownames <- function(mat) rownames(mat)
    .get_rownames_dge <- function(dge) rownames(dge$counts)
    '''
    r_pkg = STAP(r_string, "r_pkg")

    rmat = _py_to_r(mat.T)
    print(type(rmat))
    print("after conversion")
    print(type(rmat))
    rmat = r_pkg.assign_colnames(rmat, obsnames)
    print("after obsnames")
    print(type(rmat))
    rmat = r_pkg.assign_rownames(rmat, varnames)
    print("after varnames")
    print(type(rmat))
    r_pkg.print_dimnames(rmat)
    print(type(rmat))
    

    return rmat

def _ad_to_dge(adata):
    dge = edger.DGEList(
        counts = _ad_to_rmat(adata),
        samples = _py_to_r(adata.obs)
    )

    return dge