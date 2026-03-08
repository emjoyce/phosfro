from __future__ import annotations

from typing import Optional, Tuple, Any

import numpy as np
import tensorflow as tf



def total_variation_loss(img_batch: tf.Tensor) -> tf.Tensor:
    """
    Simple TV loss to encourage smoothness.

    Parameters
    ----------
    img_batch : tf.Tensor
        Tensor of shape [B, H, W, C].

    Returns
    -------
    tf.Tensor
        Scalar TV loss.
    """
    dx = img_batch[:, :, 1:, :] - img_batch[:, :, :-1, :]
    dy = img_batch[:, 1:, :, :] - img_batch[:, :-1, :, :]
    return tf.reduce_mean(tf.abs(dx)) + tf.reduce_mean(tf.abs(dy))


def ij_to_flat_index(grid: Any, i: int, j: int) -> int:
    """Return the flat (row-major) index for a given (i, j) in a 2D grid.

    This is a *standalone* helper that assumes ``grid`` is a 2D array-like
    object (NumPy array or Tensor) laid out in row-major order. The returned
    index corresponds to the position that ``grid[i, j]`` would occupy in
    ``grid.reshape(-1)``.

    Parameters
    ----------
    grid : array-like
        2D array or tensor with shape (H, W).
    i, j : int
        Row and column indices into ``grid``.

    Returns
    -------
    int
        Flat index into the row-major flattened grid.
    """
    arr = np.asarray(grid)
    if arr.ndim != 2:
        raise ValueError(f"ij_to_flat_index expects a 2D grid, got shape {arr.shape}")

    H, W = arr.shape
    if not (0 <= i < H and 0 <= j < W):
        raise IndexError(f"(i, j)=({i}, {j}) out of bounds for grid shape {arr.shape}")

    return int(np.ravel_multi_index((i, j), arr.shape))


def grid_to_flat_indices(mosaic_grid: np.ndarray, ij):
    """
    Map (i,j) grid coords -> index into the flattened 'cell_outputs' tensor
    produced by iterating over np.argwhere(mosaic_grid != -1).

    Parameters
    ----------
    mosaic_grid : np.ndarray
        2D array with -1 in invalid locations.
    ij : tuple[int,int] or array-like of shape (K,2)
        Either a single (i,j) or a list/array of (i,j) pairs.

    Returns
    -------
    int or np.ndarray
        If input is a single (i,j), returns an int index.
        If input is multiple (i,j), returns a (K,) int array of indices.

    Raises
    ------
    ValueError
        If any provided (i,j) is invalid (grid == -1) or out of bounds.
    """
    grid = mosaic_grid
    if grid.ndim != 2:
        raise ValueError(f"mosaic_grid must be 2D, got shape {grid.shape}")

    # Build lookup: grid position -> flattened tensor index
    valid_coords = np.argwhere(grid != -1).astype(int)  
    lookup = np.full(grid.shape, -1, dtype=np.int64)
    lookup[valid_coords[:, 0], valid_coords[:, 1]] = np.arange(valid_coords.shape[0], dtype=np.int64)

    # Normalize ij input to (K,2)
    ij_arr = np.asarray(ij, dtype=int)
    single = (ij_arr.shape == (2,))
    if single:
        ij_arr = ij_arr[None, :]
    if ij_arr.ndim != 2 or ij_arr.shape[1] != 2:
        raise ValueError(f"`ij` must be (2,) or (K,2); got shape {ij_arr.shape}")

    H, W = grid.shape
    i = ij_arr[:, 0]
    j = ij_arr[:, 1]

    # Bounds check
    if np.any(i < 0) or np.any(i >= H) or np.any(j < 0) or np.any(j >= W):
        bad = ij_arr[(i < 0) | (i >= H) | (j < 0) | (j >= W)]
        raise ValueError(f"One or more (i,j) are out of bounds for grid shape {grid.shape}: {bad.tolist()}")

    idx = lookup[i, j]
    if np.any(idx == -1):
        bad = ij_arr[idx == -1]
        raise ValueError(f"One or more (i,j) refer to invalid cells (grid == -1): {bad.tolist()}")

    return int(idx[0]) if single else idx

class Oracle:
    """
    Minimal oracle that optimizes an image to match a target bipolar mosaic response.

    Notes
    -----
    - Assumes images are in [0, 1] and uses clipping + stop_gradient clamping
      exactly as in the provided code.
    - Assumes the bipolar processor exposes:
        - process_new_image(image=..., method=..., stimulation_mosaic=..., amacrine_sigma_blur=...)
        - cell_outputs attribute after processing
    """

    def __init__(self, bip_processor: Any, height: int = 126, width: int = 126):
        self.bip_processor = bip_processor
        self.height = int(height)
        self.width = int(width)

    def solve(
        self,
        target_outputs: Array,
        *,
        n_steps: int = 300,
        lr: float = 1e-2,
        lambda_tv: Optional[float] = 1e-4,
        lambda_l2: float = 1e-4,
        init_image: Optional[Array] = None,
        print_every: int = 50, 
        return_loss: bool = False,
        stop_loss_threshold: Optional[float] = None,
        stop_if_no_improve_steps: Optional[int] = None,
        stop_min_improvement: float = 0.0,
    ) -> Tuple[Array, Array] | Tuple[Array, Array, Array]:
        """
        Optimize an image so that the model-predicted mosaic response matches target_outputs.

        Parameters
        ----------
        target_outputs : np.ndarray
            Target mosaic response vector (shape: [N,] or compatible).
        n_steps : int
            Number of Adam steps.
        lr : float
            Learning rate for Adam.
        lambda_tv : float | None
            Weight of TV regularization. If None or <= 0, TV loss is disabled.
        lambda_l2 : float
            Weight of L2 regularization toward mid-gray (0.5).
        init_image : np.ndarray | None
            Warm-start image of shape (H, W, 3). If None, initializes to 0.5 gray.
        print_every : int
            Print progress every N steps. Set to 0 to disable.
        return_loss : bool
            If True, also return the per-step loss values for convergence plotting.
        stop_loss_threshold : float | None
            If set, stop optimization early if loss falls below this threshold.
        stop_if_no_improve_steps : int | None
            If set, stop early when the loss has not improved by at least
            stop_min_improvement for this many consecutive steps.
        stop_min_improvement : float
            Minimum decrease in loss to count as an "improvement" when using
            stop_if_no_improve_steps. Defaults to 0.0 (any decrease).

        Returns
        -------
        If return_loss is False:
            (img_final, achieved_outputs) : (np.ndarray, np.ndarray)
                img_final has shape (H, W, 3) in [0, 1].
                achieved_outputs is the bipolar mosaic response vector achieved by img_final.

        If return_loss is True:
            (img_final, achieved_outputs, loss_history) : (np.ndarray, np.ndarray, np.ndarray)
                loss_history is a 1D array of loss values over optimization steps.
        """
        H, W = self.height, self.width
        n_steps = int(n_steps)
        lr = float(lr)
        lambda_l2 = float(lambda_l2)
        lambda_tv_val = None if lambda_tv is None else float(lambda_tv)

        if init_image is None:
            init_image = np.full((H, W, 3), 0.5, dtype=np.float32)
        else:
            init_image = np.asarray(init_image, dtype=np.float32)
            if init_image.shape != (H, W, 3):
                raise ValueError(f"init_image must have shape {(H, W, 3)}, got {init_image.shape}")

        # new guess image starts as warm start image or whatever
        img_var = tf.Variable(init_image, dtype=tf.float32)
        opt = tf.keras.optimizers.Adam(learning_rate=lr)

        # ensure mosaic response target is float32
        target_outputs_tf = tf.cast(target_outputs, tf.float32)

        loss_history = []
        data_loss_history = []
        best_loss = np.inf
        steps_since_best = 0

        for step in range(n_steps):
            with tf.GradientTape() as tape:
                img_clipped = tf.clip_by_value(img_var, 0.0, 1.0)

                # try this to fix potential gradient flaw (kept exactly)
                img_clamped = img_var + tf.stop_gradient(img_clipped - img_var)

                # calculate mosaic response to the current image
                self.bip_processor.process_new_image(
                    image=img_clamped,
                    method="grayscale",
                    stimulation_mosaic=None,
                    amacrine_sigma_blur=None,
                )

                # make sure current_outputs is also float32
                current_outputs = tf.cast(self.bip_processor.cell_outputs, tf.float32)

                # calculate loss
                data_loss = tf.reduce_mean(tf.square(current_outputs - target_outputs_tf))

                # regularizers
                tv = tf.constant(0.0, dtype=tf.float32)
                if lambda_tv_val is not None and lambda_tv_val > 0.0:
                    img_batch = img_clamped[tf.newaxis, ...]
                    tv = total_variation_loss(img_batch)

                l2 = tf.reduce_mean(tf.square(img_clamped - 0.5))

                # total loss
                loss = data_loss + (lambda_tv_val or 0.0) * tv + lambda_l2 * l2
                # early stop if absolute threshold reached
                if stop_loss_threshold is not None and loss <= stop_loss_threshold:
                    if print_every:
                        print(f"Stopping early at step {step} with loss {loss.numpy():.6f} below threshold {stop_loss_threshold:.6f}")
                    break

            # get new guess image
            grads = tape.gradient(loss, img_var)
            opt.apply_gradients([(grads, img_var)])

            # record loss for convergence plotting
            current_loss = float(loss.numpy())
            if return_loss:
                loss_history.append(current_loss)
                data_loss_history.append(data_loss.numpy())

            # early stop if no sufficient improvement for a number of steps
            if stop_if_no_improve_steps is not None:
                if current_loss < best_loss - float(stop_min_improvement):
                    best_loss = current_loss
                    steps_since_best = 0
                else:
                    steps_since_best += 1

                if steps_since_best >= int(stop_if_no_improve_steps):
                    if print_every:
                        print(
                            f"Stopping early at step {step} after "
                            f"{steps_since_best} steps without at least "
                            f"{stop_min_improvement:g} improvement in loss "
                            f"with {data_loss.numpy():.6f} loss"
                        )
                    break

            if print_every and (step % print_every == 0 or step == n_steps - 1):
                print(
                    f"step {step:4d}  loss={loss.numpy():.6f}  data={data_loss.numpy():.6f}"
                )

        # Final forward pass to return the closest mosaic response we could find to the target
        img_final = tf.clip_by_value(img_var, 0.0, 1.0)
        self.bip_processor.process_new_image(
            image=img_final,
            method="grayscale",
            stimulation_mosaic=None,
            amacrine_sigma_blur=None,
        )
        achieved_outputs = tf.cast(self.bip_processor.cell_outputs, tf.float32)

        if return_loss:
            return img_final.numpy(), achieved_outputs.numpy(), np.asarray(loss_history, dtype=np.float32), np.asarray(data_loss_history, dtype=np.float32)
        else:
            return img_final.numpy(), achieved_outputs.numpy()


    def maximize_cell(
        self,
        i: int,
        j: int,
        *,
        n_steps: int = 300,
        lr: float = 1e-2,
        lambda_tv: Optional[float] = 1e-4,
        lambda_l2: float = 1e-4,
        init_image: Optional[Array] = None,
        print_every: int = 50,
        return_loss: bool = False,
        stop_loss_threshold: Optional[float] = None,
        stop_if_no_improve_steps: Optional[int] = None,
        stop_min_improvement: float = 0.0,
    ) -> Tuple[Array, Array] | Tuple[Array, Array, Array]:
        """Optimize an image to *maximize* the response of a single mosaic cell.

        This follows the same structure as :meth:`solve`, but instead of
        matching a full target response vector, it maximizes one chosen cell's
        output (by minimizing the negative of that cell's value plus
        regularization terms).

        Parameters
        ----------
        i, j : int or array-like of int
            Row and column indices into ``bip_processor.grid_outputs`` that
            identify which mosaic cell(s) to maximize. Either single ints
            (one cell) or broadcastable 1D index arrays of the same shape
            (multiple cells). The mapping from (i, j) to the flattened
            "valid-cell" tensor follows the same convention as
            ``grid_to_flat_indices``: valid cells are traversed in
            row-major order, skipping positions where the grid is -1, and
            the mean of the selected cells' values is maximized.
        n_steps : int
            Number of Adam steps.
        lr : float
            Learning rate for Adam.
        lambda_tv : float | None
            Weight of TV regularization. If None or <= 0, TV loss is disabled.
        lambda_l2 : float
            Weight of L2 regularization toward mid-gray (0.5).
        init_image : np.ndarray | None
            Warm-start image of shape (H, W, 3). If None, initializes to 0.5 gray.
        print_every : int
            Print progress every N steps. Set to 0 to disable.
        return_loss : bool
            If True, also return the per-step loss values for convergence plotting
            (here, the scalar objective being minimized: -cell_value + regs).
        stop_loss_threshold : float | None
            If set, stop optimization early if loss falls below this threshold.
        stop_if_no_improve_steps : int | None
            If set, stop early when the loss has not improved by at least
            stop_min_improvement for this many consecutive steps.
        stop_min_improvement : float
            Minimum decrease in loss to count as an "improvement" when using
            stop_if_no_improve_steps. Defaults to 0.0 (any decrease).

        Returns
        -------
        If return_loss is False:
            (img_final, achieved_outputs) : (np.ndarray, np.ndarray)
                img_final has shape (H, W, 3) in [0, 1].
                achieved_outputs is the bipolar mosaic response vector achieved by img_final.

        If return_loss is True:
            (img_final, achieved_outputs, loss_history) : (np.ndarray, np.ndarray, np.ndarray)
                loss_history is a 1D array of objective values over optimization steps.
        """
        H, W = self.height, self.width
        n_steps = int(n_steps)
        lr = float(lr)
        lambda_l2 = float(lambda_l2)
        lambda_tv_val = None if lambda_tv is None else float(lambda_tv)

        if init_image is None:
            init_image = np.full((H, W, 3), 0.5, dtype=np.float32)
        else:
            init_image = np.asarray(init_image, dtype=np.float32)
            if init_image.shape != (H, W, 3):
                raise ValueError(f"init_image must have shape {(H, W, 3)}, got {init_image.shape}")

        img_var = tf.Variable(init_image, dtype=tf.float32)
        opt = tf.keras.optimizers.Adam(learning_rate=lr)

        loss_history = []
        best_loss = np.inf
        steps_since_best = 0

        for step in range(n_steps):
            with tf.GradientTape() as tape:
                img_clipped = tf.clip_by_value(img_var, 0.0, 1.0)
                img_clamped = img_var + tf.stop_gradient(img_clipped - img_var)

                # Forward through bipolar processor
                self.bip_processor.process_new_image(
                    image=img_clamped,
                    method="grayscale",
                    stimulation_mosaic=None,
                    amacrine_sigma_blur=None,
                )

                # Current full mosaic response flattened
                current_outputs = tf.cast(self.bip_processor.cell_outputs, tf.float32)
                outputs_flat = tf.reshape(current_outputs, [-1])

                # Use the helper to map one or more (i, j) positions in the
                # grid to flattened indices into outputs_flat, then average
                # the corresponding cell values.
                grid = np.asarray(self.bip_processor.grid_outputs)

                # Support either single ints or array-like i, j.
                if np.isscalar(i) and np.isscalar(j):
                    idx = grid_to_flat_indices(grid, (int(i), int(j)))
                    flat_indices_np = np.asarray([idx], dtype=np.int64)
                else:
                    i_arr = np.asarray(i, dtype=int)
                    j_arr = np.asarray(j, dtype=int)
                    if i_arr.shape != j_arr.shape:
                        raise ValueError(
                            f"i and j must have the same shape; got {i_arr.shape} vs {j_arr.shape}"
                        )
                    ij_pairs = np.stack([i_arr.ravel(), j_arr.ravel()], axis=1)
                    flat_indices_np = grid_to_flat_indices(grid, ij_pairs)

                flat_indices = tf.convert_to_tensor(flat_indices_np, dtype=tf.int32)
                cell_values = tf.gather(outputs_flat, flat_indices)
                cell_value = tf.reduce_mean(cell_values)

                # Regularizers
                tv = tf.constant(0.0, dtype=tf.float32)
                if lambda_tv_val is not None and lambda_tv_val > 0.0:
                    img_batch = img_clamped[tf.newaxis, ...]
                    tv = total_variation_loss(img_batch)

                l2 = tf.reduce_mean(tf.square(img_clamped - 0.5))

                # Objective: maximize cell_value -> minimize -cell_value + regs
                loss = -cell_value + (lambda_tv_val or 0.0) * tv + lambda_l2 * l2

                # early stop if absolute threshold reached
                if stop_loss_threshold is not None and loss <= stop_loss_threshold:
                    if print_every:
                        print(
                            f"Stopping early at step {step} with loss {loss.numpy():.6f} "
                            f"below threshold {stop_loss_threshold:.6f} "
                            f"and cell_value={cell_value.numpy():.6f}"
                        )
                    break

            grads = tape.gradient(loss, img_var)
            opt.apply_gradients([(grads, img_var)])

            current_loss = float(loss.numpy())
            if return_loss:
                loss_history.append(current_loss)

            # early stop if no sufficient improvement for a number of steps
            if stop_if_no_improve_steps is not None:
                if current_loss < best_loss - float(stop_min_improvement):
                    best_loss = current_loss
                    steps_since_best = 0
                else:
                    steps_since_best += 1

                if steps_since_best >= int(stop_if_no_improve_steps):
                    if print_every:
                        print(
                            f"Stopping early at step {step} after "
                            f"{steps_since_best} steps without at least "
                            f"{stop_min_improvement:g} improvement in loss "
                            f"(cell_value={cell_value.numpy():.6f})"
                        )
                    break

            if print_every and (step % print_every == 0 or step == n_steps - 1):
                print(
                    f"step {step:4d}  loss={loss.numpy():.6f}  cell_value={cell_value.numpy():.6f}"
                )

        # Final forward pass to return mosaic response from the maximized image
        img_final = tf.clip_by_value(img_var, 0.0, 1.0)
        self.bip_processor.process_new_image(
            image=img_final,
            method="grayscale",
            stimulation_mosaic=None,
            amacrine_sigma_blur=None,
        )
        achieved_outputs = tf.cast(self.bip_processor.cell_outputs, tf.float32)

        if return_loss:
            return img_final.numpy(), achieved_outputs.numpy(), np.asarray(loss_history, dtype=np.float32)
        else:
            return img_final.numpy(), achieved_outputs.numpy()