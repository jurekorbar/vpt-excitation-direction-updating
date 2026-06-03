import numpy as np
from tqdm.notebook import tqdm
import scipy


def dir_pos_error(
    df,
    dir_error=np.pi / 9,
    pos_error=0,
    grouping=None,
    true_dir_normal_to_surface=False,
):
    """Apply random direction and position errors to channel/impact data.

    Generates perturbed direction and position vectors for rows in ``df``.

    Parameters
    ----------
    df : pandas.DataFrame
        Input table containing at least the columns ``Direction_1``,
        ``Direction_2``, ``Direction_3`` and ``Position_1``, ``Position_2``,
        ``Position_3``. The function returns modified copies of this frame.
    dir_error : float, optional
        Angular error span in radians. Errors are sampled uniformly within a
        cone of half-angle ``dir_error`` around the original direction.
        Default is ``np.pi/9``.
    pos_error : float, optional
        Maximum positional perturbation applied (in the same units as the
        input positions). Samples are drawn uniformly in [-pos_error, pos_error]
        for each position component before projecting out the component along
        the direction vector. Default is 0 (no position error).
    grouping : scalar or None, optional
        If provided, only rows where ``df['Grouping'] == grouping`` are
        perturbed. If ``None`` (default), all rows are considered.
    true_dir_normal_to_surface : bool, optional
        If True, the returned ``df_err`` will contain the perturbed
        directions (so the perturbed directions are treated as the true
        surface-normal directions). If False (default), the perturbed
        directions are returned in ``df_true`` while ``df_err`` keeps the
        original directions.

    Returns
    -------
    df_true : pandas.DataFrame
        Copy of ``df`` with the (possibly) perturbed directions stored in the
        direction columns when ``true_dir_normal_to_surface`` is False.
    df_err : pandas.DataFrame
        Copy of ``df`` with perturbed positions and, depending on
        ``true_dir_normal_to_surface``, perturbed directions.
    """

    dir_cols = ['Direction_1', 'Direction_2', 'Direction_3']
    pos_cols = ['Position_1', 'Position_2', 'Position_3']

    if grouping is None:
        bool_arr = np.array([True for i in range(len(df))])
    else:
        bool_arr = np.array(df['Grouping'] == grouping)

    dir_all = np.asarray(df[dir_cols], dtype=float)
    pos_all = np.asarray(df[pos_cols], dtype=float)

    dir_original = dir_all[bool_arr]
    pos_original = pos_all[bool_arr]

    # Direction error
    dir_all_err = _generate_dir_err(dir_all, dir_error, bool_arr)
    # dir_err__ = (np.random.rand(*dir_original.shape) * 2 - 1) * dir_error
    # dir_err_ = np.cos(np.arccos(dir_original) + dir_err__)
    # dir_err = dir_err_ / np.sqrt(np.sum(dir_err_**2, axis=-1))[:, None]

    # dir_all_err = dir_all
    # dir_all_err[bool_arr] = dir_err

    # Position error
    pos_err__ = (np.random.rand(*pos_original.shape) * 2 - 1) * pos_error
    pos_err_ = pos_original + pos_err__
    pos_err_corr = np.sum(pos_err__ * dir_original, axis=-1)[:, None]
    pos_err = pos_err_ - dir_original * pos_err_corr

    pos_all_err = pos_all.copy()
    pos_all_err[bool_arr] = pos_err

    # DataFrame with new positions and directions
    df_err = df.copy()
    df_true = df.copy()

    df_err[pos_cols] = pos_all_err

    if true_dir_normal_to_surface:
        df_err[dir_cols] = dir_all_err
    else:
        df_true[dir_cols] = dir_all_err

    return df_true, df_err


def _generate_dir_err(
    dir_true: np.ndarray, dir_span: int | float, bool_err: np.ndarray
):
    """Generate perturbed direction vectors within a cone around true dirs.

    Parameters
    ----------
    dir_true : ndarray, shape (..., 3)
        Array of unit direction vectors.
    dir_span : int or float
        Half-angle of the cone (in radians) within which to sample new
        directions for the rows indicated by ``bool_err``.
    bool_err : ndarray of bool, shape (N,) or compatible
        Boolean mask selecting which rows in ``dir_true`` should be
        perturbed. Rows where ``bool_err`` is False are copied through.

    Returns
    -------
    dir_err : ndarray, shape like ``dir_true``
        Array with the same shape as ``dir_true`` where selected rows have
        been replaced by randomly sampled directions within the cone around
        the corresponding ``dir_true`` vectors.
    """

    N = dir_true[bool_err].shape[0]

    phi_samples = np.random.uniform(0, 2 * np.pi, size=N)
    u = np.random.uniform(np.cos(dir_span), 1.0, size=N)
    theta_samples = np.arccos(u)

    dir_err_ = dir_true.copy()  # np.repeat(dir_true[None], M, axis=0)
    dir_err_[bool_err, 0] = np.cos(theta_samples)
    dir_err_[bool_err, 1] = np.sin(theta_samples) * np.cos(phi_samples)
    dir_err_[bool_err, 2] = np.sin(theta_samples) * np.sin(phi_samples)

    dir_csys = _dir_csys(dir_true[bool_err])
    dir_err_[bool_err] = np.einsum('kij,kj->ki', dir_csys, dir_err_[bool_err])
    dir_err = dir_err_
    return dir_err


def _dir_csys(dir_arr):
    """Construct a local coordinate system for each direction vector.

    Parameters
    ----------
    dir_arr : ndarray, shape (M, 3)
        Array of unit direction vectors. Each row defines the local x-axis
        for a right-handed coordinate system.

    Returns
    -------
    dir_csys : ndarray, shape (M, 3, 3)
        For each input vector the returned array contains a 3x3 matrix with
        the first column equal to the input direction and the remaining two
        columns forming an orthonormal basis spanning the plane orthogonal
        to the direction.
    """

    dir_csys = np.empty((*dir_arr.shape, 3))
    dir_csys[..., 0] = dir_arr
    for i, dir_arr_i in enumerate(dir_arr):
        ns = scipy.linalg.null_space(dir_arr_i[None])
        dir_csys[i, :, 1:] = ns
    return dir_csys


def svd_pinv(a, trunc=None):
    """Compute the Moore–Penrose pseudo-inverse of an array using SVD.

    Parameters
    ----------
    a : ndarray
        Input array of shape (..., M, N) to be pseudo-inverted. Supports
        batched inputs with leading dimensions.
    trunc : int or None, optional
        If an int is provided, the smallest ``trunc`` singular values are
        zeroed before computing the inverse (useful for regularisation).
        If ``None`` (default), no truncation is applied.

    Returns
    -------
    a_inv : ndarray
        Pseudo-inverse of ``a`` with shape (..., N, M).

    Raises
    ------
    Exception
        If ``trunc`` is neither ``None`` nor an ``int``.
    """
    U, s, Vh = np.linalg.svd(a, full_matrices=False)
    if trunc is None:
        pass
    elif isinstance(trunc, int):
        s[..., -trunc:] = 0.0
    else:
        raise Exception("`trunc` must be int or None")
    s_mask = np.where(s > 0, 0, 1)
    s_inv = 1.0 / (s + s_mask) - s_mask
    a_inv = np.swapaxes(Vh.conj(), -2, -1) @ (
        s_inv[..., None] * np.swapaxes(U.conj(), -2, -1)
    )
    return a_inv


def build_Ru(chn_rel_pos, chn_dir=None, return_both=False):
    """Construct the rigid displacement influence matrix Ru (and tensor).

    This constructs the rigid-body displacement mapping used by the VPT
    solver. The function can return either the full tensor ``_Ru`` (shape
    (..., 3, 6)) or a projected Ru matrix when channel directions are
    provided.

    Parameters
    ----------
    chn_rel_pos : ndarray, shape (..., 3)
        Channel positions relative to the virtual point. The last axis must
        index x,y,z coordinates.
    chn_dir : ndarray or None, optional
        If provided with shape (..., 3) this is treated as the channel local
        direction(s) (unit vectors). When provided the result is projected
        onto these directions to produce a Ru with shape (..., 3, 6).
    return_both : bool, optional
        If True, return a tuple ``(_Ru, Ru)`` where ``_Ru`` is the full
        tensor (unprojected) and ``Ru`` is the projected matrix. If False
        (default) return only the projected result when ``chn_dir`` is
        provided, otherwise return the tensor.

    Returns
    -------
    _Ru or Ru or (_Ru, Ru) : ndarray or tuple
        Depending on inputs and ``return_both``, returns the unprojected
        tensor ``_Ru`` (shape (..., 3, 6)), the projected matrix ``Ru``
        (shape (..., 3, 6)) or a tuple of both.
    """
    rx = chn_rel_pos[..., 0]
    ry = chn_rel_pos[..., 1]
    rz = chn_rel_pos[..., 2]

    _Ru = np.zeros((*chn_rel_pos.shape[:-1], 3, 6))
    _Ru[..., :3, :3] = np.eye(3)
    _Ru[..., 0, 4] = rz
    _Ru[..., 0, 5] = -ry
    _Ru[..., 1, 5] = rx
    _Ru[..., :3, 3:6] = _Ru[..., :3, 3:6] - _Ru[..., :3, 3:6].swapaxes(-2, -1)

    if isinstance(chn_dir, np.ndarray):
        e = chn_dir[..., None, :]
        Ru = (e @ _Ru)[..., 0, :]
        return (_Ru, Ru) if return_both else Ru
    return _Ru


def build_Rf_T(imp_rel_pos, imp_dir=None, return_both=False):
    """Construct the rigid-force influence tensor _Rf_T (or projected Rf^T).

    Parameters
    ----------
    imp_rel_pos : ndarray, shape (..., 3)
        Impact positions relative to the virtual point (x,y,z coordinates
        along the last axis).
    imp_dir : ndarray or None, optional
        If provided with shape (..., 3) the tensor is projected onto the
        impact directions to produce a matrix appropriate for mapping force
        vectors to moments. If ``None`` (default) the unprojected tensor is
        returned.
    return_both : bool, optional
        If True and ``imp_dir`` is provided, return a tuple
        ``(_Rf_T, Rf_T)`` otherwise return either ``_Rf_T`` or the
        projected ``Rf_T`` depending on inputs.

    Returns
    -------
    _Rf_T or Rf_T or (_Rf_T, Rf_T) : ndarray or tuple
        The unprojected tensor with shape (..., 6, 3), the projected matrix
        (shape (..., 6, 3) or (..., 3, 6) after minor transposition in the
        function), or both when ``return_both`` is True.
    """
    rx = imp_rel_pos[..., 0]
    ry = imp_rel_pos[..., 1]
    rz = imp_rel_pos[..., 2]

    _Rf_T = np.zeros((*imp_rel_pos.shape[:-1], 6, 3))
    _Rf_T[..., :3, :3] = np.eye(3)
    _Rf_T[..., 3, 1] = -rz
    _Rf_T[..., 3, 2] = ry
    _Rf_T[..., 4, 2] = -rx
    _Rf_T[..., 3:6, :3] = _Rf_T[..., 3:6, :3] - _Rf_T[..., 3:6, :3].swapaxes(
        -2, -1
    )

    if isinstance(imp_dir, np.ndarray):
        e = imp_dir[..., None]
        Rf_T = (_Rf_T @ e)[..., 0].swapaxes(-2, -1)
        return (_Rf_T, Rf_T) if return_both else Rf_T
    return _Rf_T


def enforce_physical_Y(Y):
    """Project an FRF block onto its symmetric part to stabilise the solver.

    Parameters
    ----------
    Y : ndarray, shape (..., n, n)
        Frequency response block(s) to be symmetrised. The function returns
        0.5*(Y + Y^T) along the last two axes.

    Returns
    -------
    Y_sym : ndarray
        Symmetric projection of ``Y`` with the same shape as the input.
    """
    return 0.5 * (Y + np.swapaxes(Y, -2, -1))


def solve_impact_directions(
    Y,
    chn_rel_pos,
    chn_dir,
    imp_rel_pos,
    imp_dir,
    bool_up,
    method='average',
    maxiter=1000,
    epsilon=1e-15,
    printing=True,
    return_is_optimal=False,
    disable_tqdm=False,
):
    """Iteratively update impact/excitation direction vectors using VPT.

    Parameters
    ----------
    Y : ndarray, shape (nfreq, nch, nimp)
        FRF block for the selected channels and impacts over a set of
        frequencies.
    chn_rel_pos : ndarray, shape (nchn, 3)
        Channel positions relative to the virtual point used to build Ru.
    chn_dir : ndarray, shape (nchn, 3)
        Channel direction unit vectors (local orientations). Passed to
        ``build_Ru`` to project the Ru tensor.
    imp_rel_pos : ndarray, shape (nimp, 3)
        Impact positions relative to the virtual point.
    imp_dir : ndarray, shape (nimp, 3)
        Current impact direction vectors (will be updated for indices where
        ``bool_up`` is True).
    bool_up : ndarray of bool, shape (nimp,)
        Mask selecting which impacts in ``imp_dir`` should be updated.
    method : {'average', 'reshape', 'reshape_re_im'}, optional
        Algorithm for solving the direction update:
        - 'average' (default): sum contributions across frequencies and solve
          the smaller linear system.
        - 'reshape': reshape the problem into a larger linear system and
          solve using the real parts.
        - 'reshape_re_im': separate real and imaginary parts into a stacked
          real system before solving.
    maxiter : int, optional
        Maximum number of iterations for the fixed-point update. Default
        is 1000.
    epsilon : float, optional
        Convergence tolerance. Iteration stops when the change in the
        direction vectors (cosine overlap based) falls below ``epsilon``.
    printing : bool, optional
        If True (default) print a short convergence message when done.
    return_is_optimal : bool, optional
        If True and the solver detects an immediate convergence at the
        first iteration, return the boolean ``True`` instead of the
        directions. Note: this flag is for internal checks and typically
        the function returns the updated direction array.
    disable_tqdm : bool, optional
        If True, disable the tqdm progress bar used during iteration.

    Returns
    -------
    imp_dir_up : ndarray or bool
        If ``return_is_optimal`` is False (default) returns an array of the
        same shape as ``imp_dir`` containing the updated (unit) direction
        vectors. If ``return_is_optimal`` is True and the solver returns an
        immediate optimal indicator, the function may return ``True``.

    Raises
    ------
    ValueError
        If ``method`` is not one of the supported options.
    """
    Ru = build_Ru(chn_rel_pos, chn_dir)
    Tu = svd_pinv(Ru)
    imp_dir_up = np.copy(imp_dir)

    iter_count = -1
    for iter_count in tqdm(range(maxiter), disable=disable_tqdm):
        imp_dir_old = np.copy(imp_dir_up)
        _Rf_T, Rf_T = build_Rf_T(imp_rel_pos, imp_dir_up, return_both=True)

        Y_qm = Tu @ Y @ svd_pinv(Rf_T)
        Y_qm_sym = enforce_physical_Y(Y_qm)
        Ru_Y_qm_sym = Ru @ Y_qm_sym

        # Frequency weighting (robust to noise)
        w = 1.0 / np.max(np.max(np.abs(Ru_Y_qm_sym), axis=-1), axis=-1)
        Ru_Y_qm_sym = np.einsum('k,kij->kij', w, Ru_Y_qm_sym)
        _Y = np.einsum('k,kil->kil', w, Y)

        RuYqm_Rf_T = np.einsum('kij,mjl->mkil', Ru_Y_qm_sym, _Rf_T)

        if method == 'average':
            RuYqm_Rf_T_sum = np.sum(RuYqm_Rf_T, axis=1)
            Y_sum = np.sum(_Y, axis=0)
            Y_sum_reshaped = Y_sum.T[..., None]
            imp_dir_ = np.real(svd_pinv(RuYqm_Rf_T_sum) @ Y_sum_reshaped)[
                ..., 0
            ]
        elif method == 'reshape':
            RuYqm_Rf_T_flat = np.reshape(
                RuYqm_Rf_T, (RuYqm_Rf_T.shape[0], -1, RuYqm_Rf_T.shape[-1])
            )
            Y_flat_reshaped = np.reshape(_Y, (-1, _Y.shape[-1])).T[..., None]
            # ensure complex dtype for static type-checkers, then use real parts
            RuYqm_Rf_T_flat_c = np.asarray(
                RuYqm_Rf_T_flat, dtype=np.complex128
            )
            Y_flat_reshaped_c = np.asarray(
                Y_flat_reshaped, dtype=np.complex128
            )
            imp_dir_ = np.real(
                svd_pinv(RuYqm_Rf_T_flat_c.real) @ Y_flat_reshaped_c.real
            )[..., 0]
        elif method == 'reshape_re_im':
            RuYqm_Rf_T_flat = np.reshape(
                RuYqm_Rf_T, (RuYqm_Rf_T.shape[0], -1, RuYqm_Rf_T.shape[-1])
            )
            Y_flat_reshaped = np.reshape(_Y, (-1, _Y.shape[-1])).T[..., None]
            # cast to complex to make .real/.imag attributes visible to static checkers
            RuYqm_Rf_T_flat_c = np.asarray(
                RuYqm_Rf_T_flat, dtype=np.complex128
            )
            Y_flat_reshaped_c = np.asarray(
                Y_flat_reshaped, dtype=np.complex128
            )
            RuYqm_Rf_T_flat_re_im = np.concatenate(
                (RuYqm_Rf_T_flat_c.real, RuYqm_Rf_T_flat_c.imag), axis=-2
            )
            Y_flat_reshaped_re_im = np.concatenate(
                (Y_flat_reshaped_c.real, Y_flat_reshaped_c.imag), axis=-2
            )
            imp_dir_ = np.real(
                svd_pinv(RuYqm_Rf_T_flat_re_im) @ Y_flat_reshaped_re_im
            )[..., 0]
        else:
            raise ValueError(f"Unknown method: {method}")

        imp_dir_up_ = imp_dir_ / np.sqrt(np.sum(imp_dir_**2, axis=-1))[:, None]
        imp_dir_up[bool_up] = imp_dir_up_[bool_up]

        delta = 1.0 - np.mean(
            np.sum(imp_dir_up_[bool_up] * imp_dir_old[bool_up], axis=-1)
        )
        if epsilon > delta:
            if printing:
                print(
                    f"Impact directions converged after {iter_count} iterations."
                )
            break
        if iter_count == maxiter - 1 and printing:
            print(
                f"Impact direction convergence criterion not satisfied: ε={epsilon}>{delta}"
            )

    if iter_count == 0 and return_is_optimal:
        return True
    return imp_dir_up


def update_imp_directions_df(
    FRF,
    freq,
    df_chn,
    df_imp,
    df_vp,
    update_groups,
    ref_group=None,
    f_start=0,
    f_end=100,
    method='average',
    epsilon=1e-15,
    maxiter=1000,
    printing=True,
    scale_positions=None,
    scale_Y=None,
):
    """Update impact/excitation direction vectors in a dataframe using VPT.

    The function extracts the FRF block for the selected virtual-point group
    and reference group, runs the internal iterative solver and returns a
    copy of ``df_imp`` with updated direction columns.

    Parameters
    ----------
    FRF : ndarray, shape (nfreq, nch_total, nimp_total)
        Full FRF matrix from which blocks are selected based on group masks.
    freq : ndarray, shape (nfreq,)
        Frequencies corresponding to the first axis of ``FRF``.
    df_chn : pandas.DataFrame
        Channel dataframe containing at least columns ``Grouping`` and the
        position/direction columns used by the code.
    df_imp : pandas.DataFrame
        Impact dataframe containing at least columns ``Grouping`` and the
        position/direction columns used by the code. This frame is copied
        and a modified copy is returned.
    df_vp : pandas.DataFrame
        Virtual-point dataframe containing at least a ``Grouping`` column and
        position columns; used to determine the virtual-point location for
        the selected group.
    update_groups : scalar or sequence
        One grouping value or a list/array/tuple of grouping values to update.
        If a sequence, the function dispatches updates per-group and
        returns a combined dataframe.
    ref_group : scalar or None, optional
        Reference group to include in the solution alongside ``update_groups``.
        If ``None``, only impacts in ``update_groups`` are used.
    f_start, f_end : float, optional
        Frequency range (in same units as ``freq``) used to select the
        frequency block for the update. ``f_start`` is inclusive and
        ``f_end`` is exclusive (based on nearest-index selection).
    method : {'average', 'reshape', 'reshape_re_im'}, optional
        Solver method passed to ``solve_impact_directions``. See that
        function for available options. Default is 'average'.
    epsilon : float, optional
        Convergence tolerance forwarded to the internal solver.
    maxiter : int, optional
        Maximum iterations forwarded to the internal solver.
    printing : bool, optional
        Verbosity flag forwarded to the internal solver.
    scale_positions : float or None, optional
        Optional scaling factor applied to all position vectors (default: 1.0
        when ``None``). Useful for unit conversions.
    scale_Y : float or None, optional
        Optional scaling factor applied to the selected FRF block (default:
        1.0 when ``None``).

    Returns
    -------
    df_imp_up : pandas.DataFrame
        A copy of ``df_imp`` where the direction columns
        (``Direction_1``..``Direction_3``) have been replaced with the
        updated direction vectors for the selected impacts/groups.
    """
    # normalize inputs
    if scale_Y is None:
        scale_Y = 1.0
    if scale_positions is None:
        scale_positions = 1.0

    # support list of update groups by delegating per-group
    if isinstance(update_groups, (list, tuple, np.ndarray)):
        df_imp_new = df_imp.copy()
        for ug in update_groups:
            out = update_imp_directions_df(
                FRF,
                freq,
                df_chn,
                df_imp,
                df_vp,
                ug,
                ref_group,
                f_start,
                f_end,
                method,
                epsilon,
                maxiter,
                printing,
                scale_positions,
                scale_Y,
            )
            df_imp_new[df_imp['Grouping'] == ug] = out[
                df_imp['Grouping'] == ug
            ]
        return df_imp_new

    update_group = update_groups

    pos_cols = ['Position_1', 'Position_2', 'Position_3']
    dir_cols = ['Direction_1', 'Direction_2', 'Direction_3']

    bool_vp = np.array(df_vp['Grouping'] == update_group)
    vp_pos = np.array(df_vp[pos_cols])[bool_vp][0] * scale_positions

    bool_imp_up = np.array(df_imp['Grouping'] == update_group)
    bool_imp_ref = np.array(df_imp['Grouping'] == ref_group)
    bool_imp = bool_imp_up + bool_imp_ref

    chn_ix, imp_ix = np.ix_(
        np.array(df_chn['Grouping'] == update_group), bool_imp
    )
    start = np.argmin(np.abs(freq - f_start))
    end = np.argmin(np.abs(freq - f_end)) + 1

    # FRF block for selected channels (group) and impacts (group+ref)
    Y = FRF[start:end, chn_ix, imp_ix] * scale_Y

    # geometry and directions
    imp_pos_all = np.asarray(df_imp[pos_cols], dtype=float) * scale_positions
    imp_pos = imp_pos_all[bool_imp]

    imp_dir_all = np.asarray(df_imp[dir_cols], dtype=float)
    imp_dir = imp_dir_all[bool_imp]

    chn_pos_all = np.asarray(df_chn[pos_cols], dtype=float) * scale_positions
    chn_pos = chn_pos_all[np.array(df_chn['Grouping'] == update_group)]
    chn_dir_all = np.asarray(df_chn[dir_cols], dtype=float)
    chn_dir = chn_dir_all[np.array(df_chn['Grouping'] == update_group)]

    chn_rel_pos = chn_pos - vp_pos
    imp_rel_pos = imp_pos - vp_pos

    bool_up = np.array(df_imp['Grouping'] == update_group)[bool_imp]

    # perform direction update using the internal solver
    imp_dir_up = solve_impact_directions(
        Y,
        chn_rel_pos,
        chn_dir,
        imp_rel_pos,
        imp_dir,
        bool_up,
        method=method,
        maxiter=maxiter,
        epsilon=epsilon,
        printing=printing,
    )

    imp_dir_all_up = imp_dir_all.copy()
    imp_dir_all_up[bool_imp] = imp_dir_up
    df_imp_up = df_imp.copy()
    df_imp_up[dir_cols] = imp_dir_all_up

    return df_imp_up
