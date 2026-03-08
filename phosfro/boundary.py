import tensorflow as tf
from .oracle import Oracle

# first, the scoot code 

def chord_step_size(
    warm_start: tf.Tensor,
    target: tf.Tensor,
    n_steps: int = 30,
    eps: float = 1e-12,
) -> tf.Tensor:
    """
    Compute the Euclidean step size along the chord from warm_start to target,
    dividing the total chord length by n_steps.

    Returns a scalar tf.Tensor (float32).
    """
    warm_start = tf.cast(tf.reshape(warm_start, [-1]), tf.float32)
    target = tf.cast(tf.reshape(target, [-1]), tf.float32)

    delta = target - warm_start
    length = tf.linalg.norm(delta)  # scalar
    n_steps_f = tf.cast(tf.maximum(n_steps, 1), tf.float32)

    # Avoid zero division / NaNs if length is ~0
    step = length / n_steps_f
    return tf.maximum(step, tf.cast(eps, tf.float32))


def scoot_one_step(
    warm_start: tf.Tensor,
    target: tf.Tensor,
    step_size: tf.Tensor,
    allow_overshoot: bool = False,
) -> tf.Tensor:
    """
    Take one step from warm_start toward target with *fixed Euclidean step_size*
    along the straight line segment.

    Returns the new point (same shape as warm_start), float32 tf.Tensor.

    If warm_start == target (or nearly), returns warm_start.
    """
    warm_start_f = tf.cast(warm_start, tf.float32)
    target_f = tf.cast(target, tf.float32)

    ws_flat = tf.reshape(warm_start_f, [-1])
    t_flat = tf.reshape(target_f, [-1])

    delta = t_flat - ws_flat
    dist = tf.linalg.norm(delta)  # scalar

    # Unit direction; if dist==0, direction doesn't matter (handle via safe divide)
    direction = tf.math.divide_no_nan(delta, dist)

    new_flat = ws_flat + tf.cast(step_size, tf.float32) * direction

    # By default, clamp so we never step past the target (keeps alpha in [0,1]).
    # If allow_overshoot is True, we skip this clamp and allow steps beyond
    # the target along the same line.
    if not allow_overshoot:
        new_flat = tf.where(step_size >= dist, t_flat, new_flat)

    return tf.reshape(new_flat, tf.shape(warm_start_f))





# then, binary search code
class ChordRunner:
    """
    Runs a simple "scoot search" along the chord from a warm-start mosaic response r0
    to a target mosaic response s.

    For each chord point r_k, it calls the oracle to find an image whose response
    matches r_k, using continuation (the previous step's optimized image is the next init).

    Returned artifacts are intended for later plotting:
    - chord_targets: the intended mosaics along the chord (numpy)
    - found_images:  optimized images per step (numpy)
    - found_mosaics: achieved mosaics per step (numpy)
    - losses:        response-space MSE per step (float)
    """

    def __init__(self, oracle: Oracle):
        self.oracle = oracle

    def run(
        self,
        warm_start_image,
        warm_start_mosaic,
        target_mosaic,
        *,
        n_chord_steps: int = 30,
        oracle_steps: int = 50,
        lr: float = 1e-2,
        lambda_tv = 0.0,
        lambda_l2: float = 1e-4,
        print_every: int = 50,
        stimulation_mosaic = None,
        amacrine_sigma_blur = None,
        verbose: bool = True,
        stop_loss_threshold = None,
        stop_if_no_improve_steps = None,
        stop_min_improvement = 0.0,
    ):
        """
        Parameters
        ----------
        warm_start_image:
            Starting image x0 (H,W,3) in [0,1] (tf or numpy).
        warm_start_mosaic:
            Starting response r0 (tf or numpy).
        target_mosaic:
            Target response s (tf or numpy).
        n_chord_steps:
            Number of equal-length scoot steps from r0 toward s.
        oracle_steps:
            Number of optimization steps per chord point.
        lr, lambda_tv, lambda_l2:
            Oracle hyperparameters per step.
        print_every:
            Oracle printing cadence (0 disables).
        verbose:
            If True, prints step size and per-step progress lines.
        stop_loss_threshold:
            If set, stops oracle optimization early if loss falls below this threshold.
        stop_if_no_improve_steps:
            If set, passes a patience parameter to the oracle so it stops
            when loss has not improved by at least stop_min_improvement
            for this many consecutive steps.
        stop_min_improvement:
            Minimum loss decrease to count as an improvement for the
            stop_if_no_improve_steps criterion.

        Returns
        -------
        dict with keys:
            step_size, chord_targets, found_images, found_mosaics, losses
        """
        r0 = tf.cast(tf.reshape(tf.convert_to_tensor(warm_start_mosaic), [-1]), tf.float32)
        s = tf.cast(tf.reshape(tf.convert_to_tensor(target_mosaic), [-1]), tf.float32)

        n_chord_steps = int(n_chord_steps)
        if n_chord_steps < 1:
            raise ValueError("n_chord_steps must be >= 1")

        # Compute fixed Euclidean step size and build chord targets.
        eta = chord_step_size(r0, s, n_steps=n_chord_steps)
        if verbose:
            print(f"[ScootChordRunner] step_size (eta) = {float(eta.numpy()):.6f}")

        chord_targets_tf: List[tf.Tensor] = [r0]
        cur = r0
        for _ in range(n_chord_steps):
            cur = scoot_one_step(cur, s, eta)
            chord_targets_tf.append(cur)

        chord_targets = [t.numpy() for t in chord_targets_tf]

        # Iterate from warm start toward target, optimizing at each chord point.
        found_images: List[Array] = []
        found_mosaics: List[Array] = []
        losses: List[float] = []
        mse_losses = []

        init_img = warm_start_image

        for k in range(1, len(chord_targets_tf)):
            target_k = chord_targets_tf[k]
            if verbose:
                print(f"[ScootChordRunner] step {k}/{n_chord_steps}")

            img_k, mos_k, loss_k, mse_loss_k = self.oracle.solve(
                target_k,
                n_steps=oracle_steps,
                lr=lr,
                lambda_tv=lambda_tv,
                lambda_l2=lambda_l2,
                init_image=init_img,
                print_every=print_every,
                return_loss=True,
                stop_loss_threshold=stop_loss_threshold,
                stop_if_no_improve_steps=stop_if_no_improve_steps,
                stop_min_improvement=stop_min_improvement,

            )

            found_images.append(img_k)
            found_mosaics.append(mos_k)
            losses.append(loss_k)
            mse_losses.append(mse_loss_k)

            # continuation: next init is this step's optimized image
            init_img = img_k

        return {
            "step_size": float(eta.numpy()),
            "chord_targets": chord_targets,      # list[np.ndarray] length n_chord_steps+1
            "found_images": found_images,        # list[np.ndarray] length n_chord_steps
            "found_mosaics": found_mosaics,      # list[np.ndarray] length n_chord_steps
            "losses": losses,                    # list[float] length n_chord_steps
            "mse_losses": mse_losses,            # list[float] length n_chord_steps
        }