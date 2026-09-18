"""Figures used to inspect each processing step."""

import matplotlib.pyplot as plt
import numpy as np
from skimage import measure


def show_phase_results(results):
    """
    Display the wrapped phase and the available unwrapped phase results.
    The wrapped phase is displayed in the range [-pi, pi].
    """
    phase = results["phase"]
    phase_skimage = results["phase_unwrapped_skimage"]
    phase_custom = results["phase_unwrapped_custom"]

    images = [phase]
    titles = ["Wrapped Phase"]
    is_wrapped = [True]

    if phase_skimage is not None:
        images.append(phase_skimage)
        titles.append("Unwrapped Phase - Skimage")
        is_wrapped.append(False)

    if phase_custom is not None:
        images.append(phase_custom)
        titles.append("Unwrapped Phase - Custom")
        is_wrapped.append(False)

    # Find common limits for all available unwrapped phase images
    unwrapped_images = [
        image for image, wrapped in zip(images, is_wrapped)
        if not wrapped
    ]

    if unwrapped_images:
        unwrap_vmin = min(np.min(image) for image in unwrapped_images)
        unwrap_vmax = max(np.max(image) for image in unwrapped_images)

    n_images = len(images)

    fig, axes = plt.subplots(
        1,
        n_images,
        figsize=(5 * n_images, 4)
    )

    if n_images == 1:
        axes = [axes]

    for ax, image, title, wrapped in zip(
        axes, images, titles, is_wrapped
    ):

        if wrapped:
            im = ax.imshow(image, cmap="gray", vmin=-np.pi, vmax=np.pi)
        else:
            im = ax.imshow(image, cmap="gray", vmin=unwrap_vmin, vmax=unwrap_vmax)

        ax.set_title(title)
        ax.axis("off")

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Phase [rad]")

        if wrapped:
            cbar.set_ticks([-np.pi, 0, np.pi])
            cbar.set_ticklabels(
                [r"$-\pi$", "0", r"$\pi$"]
            )

    plt.tight_layout()
    plt.show()


def show_segmentation_results(phase_image, segmentation_result):
    """
    Display the main results of the RBC segmentation process.

    Shows:
        1. Original phase image.
        2. Final binary RBC mask.
        3. Detected RBCs with contours and labels.
    """
    phase_image = np.asarray(phase_image)

    binary_mask = segmentation_result["binary_mask"]
    labeled_mask = segmentation_result["labeled_mask"]
    num_rbcs = segmentation_result["num_rbcs"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original phase image
    im = axes[0].imshow(
        phase_image,
        cmap="gray"
    )
    axes[0].set_title("Phase Image")
    axes[0].axis("off")
    fig.colorbar(im, ax=axes[0], label="Phase [rad]")

    # Binary mask
    axes[1].imshow(
        binary_mask,
        cmap="gray"
    )
    axes[1].set_title("Binary RBC Mask")
    axes[1].axis("off")

    # Detected RBCs
    axes[2].imshow(
        phase_image,
        cmap="gray"
    )

    for label in range(1, num_rbcs + 1):

        # Individual RBC mask
        rbc_mask = labeled_mask == label

        # Find RBC contour
        contours = measure.find_contours(
            rbc_mask,
            level=0.5
        )

        for contour in contours:
            axes[2].plot(
                contour[:, 1],
                contour[:, 0],
                linewidth=1.5
            )

        # Temporary center only for displaying the RBC label
        y_coords, x_coords = np.where(rbc_mask)

        if len(x_coords) > 0:
            center_x = np.mean(x_coords)
            center_y = np.mean(y_coords)

            axes[2].text(
                center_x,
                center_y,
                str(label),
                fontsize=8,
                ha="center",
                va="center"
            )

    axes[2].set_title(
        f"Detected RBCs: {num_rbcs}"
    )
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()


def show_phase_profiles_overview(phase_image, labeled_mask, profile_result):
    """
    Display the phase image with RBC contours, RBC IDs, and the two
    phase-profile lines used for each valid RBC.

    outputs: matplotlib.figure.Figure Overview figure.
    """
    phase_image = np.asarray(phase_image)
    labeled_mask = np.asarray(labeled_mask)
    rbcs = (profile_result["rbcs"])
    num_excluded = (profile_result["num_excluded"])
    regions = measure.regionprops(labeled_mask)

    # Create figure
    fig, ax = plt.subplots(figsize=(9, 9))
    image = ax.imshow(phase_image, cmap="viridis")

    # Draw RBC contours and IDs
    for region in regions:
        label = int(region.label)
        contours = measure.find_contours(labeled_mask == label, 0.5)

        for contour in contours:
            ax.plot(contour[:, 1], contour[:, 0], linewidth=0.8)

        cy, cx = (
            region.centroid)
        ax.text(
            cx, cy, str(label), fontsize=7, ha="center", va="center",
            bbox={"boxstyle": "round,pad=0.15", "alpha": 0.5}
        )

    # Draw the two phase-profile lines
    for rbc in rbcs:
        profile_1 = (rbc["profile_1"])
        profile_2 = (rbc["profile_2"])
        coords_1 = (profile_1["coords_rc"])
        coords_2 = (profile_2["coords_rc"])

        # Profile 1
        ax.plot(coords_1[:, 1], coords_1[:, 0], linewidth=1.2)

        # Profile 2
        ax.plot(coords_2[:, 1], coords_2[:, 0], linewidth=1.2)

    # Figure settings
    ax.set_title("Phase Profile Measurements\n" f"{profile_result['num_rbcs']} RBCs analyzed, " f"{num_excluded}excluded")
    ax.axis("off")

    fig.colorbar(image, ax=ax, label="Phase [rad]", fraction=0.046)
    fig.tight_layout()
    plt.show()

    return fig


def plot_phase_profiles(profile_result, profiles_per_figure=20):
    """
    Display the phase profiles of all valid RBCs in batches.
    Each subplot shows: Profile 1, Profile 2
        - Maximum and minimum of both Profiles, High- and low-region mean phase values
        - delta_phase_1, delta_phase_2, Mean delta_phase

    outcomes:
         List containing the generated matplotlib Figure objects.
    """
    if profiles_per_figure < 1:
        raise ValueError("profiles_per_figure must be greater than zero.")

    rbcs = (profile_result["rbcs"])
    figures = []

    if len(rbcs) == 0:
        return figures

    # Number of batches
    num_batches = int(np.ceil(len(rbcs) / profiles_per_figure))

    # Generate each batch
    for batch_idx in range(num_batches):
        start = (batch_idx * profiles_per_figure)
        end = min(start + profiles_per_figure, len(rbcs))
        batch_rbcs = (rbcs[start:end])

        # Figure organization
        ncols = 4
        nrows = int(np.ceil(len(batch_rbcs) / ncols))

        fig, axes = plt.subplots(nrows, ncols, figsize=(16, 3.7 * nrows), squeeze=False)
        axes = (axes.flatten())

        # Plot each RBC
        for ax, rbc in zip(axes, batch_rbcs):
            p1 = (rbc["profile_1"])
            p2 = (rbc["profile_2"])

            # Plot both phase profiles
            line_1, = ax.plot(p1["distance_um"], p1["phase"], label="Profile 1")
            line_2, = ax.plot(p2["distance_um"], p2["phase"], label="Profile 2")

            # Find extrema positions - Profile 1
            idx_max_1 = int(np.argmax(p1["phase"]))
            idx_min_1 = int(np.argmin(p1["phase"]))

            # Maximum - Profile 1
            ax.plot(p1["distance_um"][idx_max_1], p1["phase"][idx_max_1], "o", color=line_1.get_color())

            # Minimum - Profile 1
            ax.plot(p1["distance_um"][idx_min_1], p1["phase"][idx_min_1], "v", color=line_1.get_color())

            # Find extrema positions - Profile 2
            idx_max_2 = int(np.argmax(p2["phase"]))
            idx_min_2 = int(np.argmin(p2["phase"]))

            # Maximum - Profile 2
            ax.plot(p2["distance_um"][idx_max_2], p2["phase"][idx_max_2], "o", color=line_2.get_color())

            # Minimum - Profile 2
            ax.plot(p2["distance_um"][idx_min_2],p2["phase"][idx_min_2], "v", color=line_2.get_color())

            # Mean high- and low-phase regions - Profile 1
            ax.axhline(p1["high_region_mean_phase"], linestyle="--", linewidth=0.8, color=line_1.get_color())
            ax.axhline(p1["low_region_mean_phase"], linestyle=":", linewidth=0.8, color=line_1.get_color())

            # Mean high- and low-phase regions - Profile 2
            ax.axhline(p2["high_region_mean_phase"], linestyle="--", linewidth=0.8, color=line_2.get_color())
            ax.axhline(p2["low_region_mean_phase"], linestyle=":", linewidth=0.8, color=line_2.get_color())

            # Plot information
            ax.set_title(
                f"RBC {rbc['rbc_id']}\n"
                rf"$\Delta\phi_1$ = "
                f"{rbc['delta_phase_1']:.3f} rad | "
                rf"$\Delta\phi_2$ = "
                f"{rbc['delta_phase_2']:.3f} rad\n"
                rf"$\overline{{\Delta\phi}}$ = "
                f"{rbc['delta_phase']:.3f} rad",
                fontsize=9
            )

            ax.set_xlabel("Distance [um]")
            ax.set_ylabel("Phase [rad]")
            ax.grid(alpha=0.2)
            ax.legend(fontsize=7)

        # Hide unused axes in the last batch
        for ax in axes[len(batch_rbcs):]:
            ax.axis("off")

        # Figure title
        fig.suptitle("RBC Phase Profiles " f"({start + 1}-{end})", fontsize=14)

        fig.tight_layout(rect=[0, 0, 1, 0.97])
        figures.append(fig)

    plt.show()

    return figures
