import csv
import warnings
from importlib import resources
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import yaml
from scipy.cluster.vq import kmeans2

import optiland.backend as be
from optiland.materials.material import Material


def glasses_selection(
    lambda_min: float,
    lambda_max: float,
    catalogs: Optional[list[str]] = None,
) -> list[str]:
    """
    Retrieves a list of glass names whose tabulated transmission window fully covers
    the specified wavelength interval [lambda1, lambda2].
    Optionally filters by catalog names.

    Args:
        lambda_min (float): The lower wavelength bound in microns.
        lambda_max (float): The upper wavelength bound in microns.
        catalogs (list[str], optional): List of catalog names.
            Catalog names are case-insensitive.
            Defaults to None.

    Returns:
        list[str]: List of unique glass names transmissive within
                   the wavelength interval and matching catalogs $
                   if specified.
    """

    csv_path = resources.files("optiland.database").joinpath("catalog_nk.csv")
    catalogs = [c.lower() for c in catalogs]
    glasses = set()

    with open(csv_path, encoding="utf-8") as file:
        reader = csv.DictReader(file, delimiter=",")
        for row in reader:
            group = row["group"].lower()
            catalog = row["filename"].split("/")[1]
            try:
                min_wavelength = float(row["min_wavelength"])
                max_wavelength = float(row["max_wavelength"])
            except ValueError:
                continue  # Skip rows with invalid wavelength values

            # Check if it's a glass, within wavelength range,
            # and optionally matches catalog
            if (
                "glass" in group
                and min_wavelength <= lambda_min
                and max_wavelength >= lambda_max
                and (catalogs is None or catalog in catalogs)
            ):
                glasses.add(row["filename_no_ext"])

    return sorted(glasses)


def get_nd_vd(glass: str) -> tuple[float, float]:
    """
    Retrieve the refractive index (n_d) and Abbe number (V_d) for a given glass.

    This function loads the material data associated with the specified glass name,
    reads its YAML specification file, and extracts the n_d and V_d values from the
    "SPECS" section. If either value is missing, defaults to (0, 0).

    Args:
        glass (str): Name of the glass material.

    Returns:
        tuple[float, float]: A tuple (n_d, V_d) representing
        the refractive index and Abbe number, in the form:
        glass_dict = {'S-BSM22': (1.62, 53.16), 'LF5G19': (1.59, 39.89), ...}
    """
    material = Material(glass)
    yml_path, _ = material._retrieve_file()
    with Path(yml_path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    try:
        nd = data["SPECS"]["nd"]
        vd = data["SPECS"]["Vd"]
    except KeyError:
        nd, vd = 0, 0
    return nd, vd


def downsample_glass_map(glass_dict: dict, num_glasses_to_keep: int) -> dict:
    """
    Downsample a glass dictionary using K-Means clustering in the (n_d, V_d) space.

    The dictionary must be in the form:
        glass_dict = {'S-BSM22': (1.62, 53.16), 'LF5G19': (1.59, 39.89), ...}

    Args:
        glass_dict (dict): Dictionary with glass names as keys
                           and (n_d, V_d) tuples as values.
        num_glasses_to_keep (int): Number of representative glasses
                                   to retain after clustering.

    Returns:
        dict: Downsampled dictionary with selected representative glasses.

    Raises:
        ValueError: If num_glasses_to_keep is greater
                    than the number of available glasses.
    """
    # Validate input
    assert num_glasses_to_keep <= len(glass_dict), (
        "Cannot keep more glasses than available in the input dictionary."
    )
    assert num_glasses_to_keep > 1, "Must retain at least 2 glasses."

    # Extract glass names and (n_d, V_d) data
    glass_names = list(glass_dict.keys())
    glass_data = be.array([glass_dict[name] for name in glass_names])

    # Detach if PyTorch and convert to NumPy for SciPy clustering
    if be.get_backend() == "torch":
        glass_data = glass_data.detach().cpu().numpy()

    # Perform K-Means clustering
    centroids, labels = kmeans2(
        glass_data, num_glasses_to_keep, minit="points", seed=1234
    )

    # Warn if kmeans2 produced fewer clusters than requested
    num_unique_clusters = len(set(labels))
    if num_unique_clusters < num_glasses_to_keep:
        warnings.warn(
            f"In downsample_glass_map(): K-Means produced only "
            f"{num_unique_clusters} clusters out of "
            f"{num_glasses_to_keep} requested. "
            f"Some clusters may be empty and "
            f"fewer glasses will be selected.",
            category=UserWarning,
            stacklevel=2,
        )

    # Select the closest glass to each centroid
    selected_glasses = {}
    labels = be.array(labels)
    for cluster_index in range(num_glasses_to_keep):
        # Get indices of glasses in this cluster
        cluster_indices = be.where(labels == cluster_index)[0]

        # Extract cluster points
        cluster_points = be.array(glass_data[cluster_indices])
        centroid = be.array(centroids[cluster_index])

        # Ensure cluster_points and centroid are 2D in torch
        if be.get_backend() == "torch":
            if cluster_points.ndim == 1:
                cluster_points = cluster_points.unsqueeze(0)
            if centroid.ndim == 1:
                centroid = centroid.unsqueeze(0)
            # Compute distances
            delta = cluster_points - centroid
            distances = be.linalg.norm(delta, dim=1 if delta.ndim > 1 else 0)
            closest_local_idx = be.argmin(distances).item()
        else:
            # NumPy distances
            distances = be.linalg.norm(cluster_points - centroid, axis=1)
            closest_local_idx = be.argmin(distances)

        # Get global index of closest point
        closest_index = cluster_indices[closest_local_idx]
        glass_name = glass_names[closest_index]
        selected_glasses[glass_name] = glass_dict[glass_name]

    return selected_glasses


def get_neighbour_glasses(
    glass: str,
    glass_selection: list[str],
    num_neighbours: int = 3,
    plot: bool = False,
) -> list[str]:
    """
    Return the `num_neighbours` closest glasses
    to the given `glass` in (n_d, V_d) space.

    Args:
        glass (str): Name of the reference glass.
        glass_selection (list[str]): List of glass names to search among.
        num_neighbours (int): Number of closest glasses to return.
        plot (bool): If True, plot the selected glass map
                     and highlight neighbors.

    Returns:
        list[str]: List of `num_neighbours` closest glasses
                   to `glass` in (n_d, V_d) space.
    """
    if glass not in glass_selection:
        raise ValueError(
            f"In get_neighbor_glasses(): glass {glass}' not found in selected glasses"
        )

    # Get index and dispersion (n_d, V_d) for all glasses
    glass_dict = {g: get_nd_vd(g) for g in glass_selection}
    nd0, vd0 = glass_dict[glass]

    # Compute Euclidean distance in (nd, vd) space. Weighting can be added.
    # For each glass in the catalogue:
    # - Skip the current glass (it should not be a neighbor of itself)
    # - Calculate the Euclidean distance in (nd, vd) space
    #   between the candidate and the current glass
    # - Store the distance and the corresponding glass name
    #   in a list for sorting
    distances: list[tuple[float, str]] = []
    for g, (nd, vd) in glass_dict.items():
        if g == glass:
            continue
        w_nd, w_vd = 1.0, 1.0  # For future weighting
        distance = be.hypot(w_nd * (nd - nd0), w_vd * (vd - vd0))
        distances.append((distance, g))

    distances.sort(key=lambda t: t[0])
    neighbours: list[str] = [g for _, g in distances[:num_neighbours]]

    # Optional plotting
    if plot:
        plot_glass_map(
            glass_selection=list(glass_dict.keys()),
            highlights=neighbours,
            title="Glass map",
        )

    return neighbours


def plot_glass_map(
    glass_selection: list[str],
    highlights: list[str],
    title: str = None,
) -> None:
    """
    Plot a (n_d, V_d) scatter map of selected optical glasses,
    highlighting specific ones.

    This visualization is useful to understand the distribution
    of glasses in the (index, Abbe number) space and to inspect
    how selected or substituted glasses relate  to the rest of
    the catalogue.

    Args:
        glass_selection (list[str]): List of glass names to plot.
        highlights (list[str]): List of glass names to highlight
        in red (e.g. current neighbors).
        title (str, optional): Custom title for the plot.
        If not provided, a default title is used.

    Notes:
        - Glasses with invalid (0, 0) values for (n_d, V_d)
          are skipped and reported.
        - The V_d axis is reversed to match
          optical engineering conventions.
    """
    # Buffers for standard and highlighted glasses
    x_vd, y_nd, labels = [], [], []
    x_vd_hl, y_nd_hl, labels_hl = [], [], []

    # Retrieve (n_d, V_d) data for all selected glasses
    glass_dict = {g: get_nd_vd(g) for g in glass_selection}

    for glass_name in glass_selection:
        nd, vd = glass_dict.get(glass_name, (0, 0))

        # Skip glasses with missing or invalid data
        if (nd, vd) == (0, 0):
            # print(f"In plot_glass_map(): couldn't display "
            #       f"{glass_name}, invalid (n_d, V_d) data.")
            continue

        # Route to highlight or standard category
        if glass_name in highlights:
            x_vd_hl.append(vd)
            y_nd_hl.append(nd)
            labels_hl.append(glass_name)
        else:
            x_vd.append(vd)
            y_nd.append(nd)
            labels.append(glass_name)

    # Create plot
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot all standard glasses
    plt.scatter(x_vd, y_nd, c="gray", marker="+", s=20, label="Glasses")
    for i, glass_name in enumerate(labels):
        plt.text(x_vd[i], y_nd[i], glass_name, fontsize=8, ha="right", va="bottom")

    # Plot highlighted glasses in red
    if x_vd_hl:
        plt.scatter(x_vd_hl, y_nd_hl, c="red", label="Highlighted")
        for i, glass_name in enumerate(labels_hl):
            plt.text(
                x_vd_hl[i],
                y_nd_hl[i],
                glass_name,
                fontsize=9,
                ha="left",
                va="top",
                color="red",
                fontweight="bold",
            )

    # Set axis labels and title
    plt.xlabel("$V_d$")
    plt.ylabel("$n_d$")
    plt.title("Glass Map: $n_d$ vs. $V_d$" if title is None else title)

    # Reverse x-axis (V_d) to match optical convention
    plt.gca().invert_xaxis()
    plt.grid(alpha=0.5)

    # Alternate horizontal bands for readability
    fig.canvas.draw()
    yticks = ax.get_yticks()
    for i in range(1, len(yticks), 2):
        try:
            ax.axhspan(yticks[i], yticks[i + 1], facecolor="lightblue", alpha=0.15)
        except IndexError:
            break  # Last pair might not exist

    plt.tight_layout()
    plt.show()
