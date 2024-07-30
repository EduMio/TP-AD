from collections.abc import Callable
import pandas as pd

def generate_refinements(subgroup: dict, mincov: int, data: pd.DataFrame) -> list:
    """
    Generate refined subgroups by adding a condition to the current subgroup.

    Parameters:
    - subgroup: A dictionary representing the current subgroup's conditions.
    - mincov: The minimum coverage threshold for subgroups.
    - data: The pandas DataFrame containing the dataset.

    Returns:
    - A list of new refined subgroups.
    """
    refinements = []
    # Iterate over each attribute in the dataset
    for column in data.columns:
        if column in subgroup:
            continue  # Skip columns already in the subgroup description
        
        # Determine the attribute type and generate conditions accordingly
        for value in [True, False]:
            # Binary attribute
            new_condition = {column: value}
            refined_subgroup = {**subgroup, **new_condition}
            if len(data.query(' and '.join([f'{k}=={repr(v)}' for k, v in refined_subgroup.items()]))) >= mincov:
                refinements.append(refined_subgroup)

    return refinements

def update_top_k(R: list, j: int, candidate: dict, quality: int):
    # Maintain a top-k list of subgroups based on quality measure
    if len(R) < j:
        R.append((candidate, quality))
    else:
        # Replace the worst quality subgroup if the candidate is better
        min_quality = min(R, key=lambda x: x[1])[1]
        if quality > min_quality:
            R.remove(min(R, key=lambda x: x[1]))
            R.append((candidate, quality))

def subgroup_selection(candidates: list, quality_func: Callable[[dict, pd.DataFrame], int], params, data: pd.DataFrame) -> list:
    # Selection strategy to choose subgroups from candidates
    selected = []
    # Implement cover-based or description-based selection strategy
    for candidate in candidates:
        quality = quality_func(candidate, data)
        # Selection logic based on strategy
        selected.append((candidate, quality))
    # Sort and select top candidates based on the strategy
    selected.sort(key=lambda x: x[1], reverse=True)
    return [x[0] for x in selected[:params.get('beam_width', len(selected))]]

def apply_dominance_pruning(subgroup: tuple, quality_func: Callable[[dict, pd.DataFrame], int], data: pd.DataFrame) -> dict:
    """
    Apply dominance-based pruning to improve a subgroup's description.

    Parameters:
    - subgroup: A dictionary representing the current subgroup's conditions.
    - quality_func: A function that evaluates the quality of a subgroup.
    - data: The pandas DataFrame containing the dataset.

    Returns:
    - A pruned subgroup with unnecessary conditions removed.
    """
    # Start with the full subgroup
    current_quality = subgroup[1]
    pruned_subgroup = subgroup[0].copy()
    # Attempt to remove each condition
    if len(pruned_subgroup.keys()) == 1:
        return pruned_subgroup
    
    print("Pruning:", pruned_subgroup, current_quality)
    for condition in list(subgroup[0].keys()):
        # Create a temporary subgroup without the current condition
        temp_subgroup = pruned_subgroup.copy()
        del temp_subgroup[condition]

        # Calculate the quality of the temporary subgroup
        temp_quality = quality_func(temp_subgroup, data)

        # If the quality is the same or improved, keep the condition removed
        if temp_quality >= current_quality:
            pruned_subgroup = temp_subgroup
            current_quality = temp_quality

        if len(pruned_subgroup.keys()) == 1:
            break

    return pruned_subgroup

def remove_duplicates(subgroups: list) -> list:
    # Remove duplicate subgroups based on their descriptions
    unique_subgroups = []
    seen_descriptions = set()
    for subgroup in subgroups:
        description = str(subgroup)  # Create a hashable representation
        if description not in seen_descriptions:
            seen_descriptions.add(description)
            unique_subgroups.append(subgroup)
    return unique_subgroups

def dssd(S: pd.DataFrame, ϕ: Callable[[dict, pd.DataFrame], int], j: int, k: int, mincov: int, maxdepth: int, P: dict) -> list:
    R = []
    Beam = [{}]  # Start with an empty subgroup
    depth = 1

    while depth <= maxdepth:
        Cands = []
        for b in Beam:
            Cands.extend(generate_refinements(b, mincov, S))

        for c in Cands:
            update_top_k(R, j, c, ϕ(c, S))

        Beam = subgroup_selection(Cands, ϕ, P, S)
        depth += 1
        print("Depth:", depth)

    # Dominance pruning and removing duplicates
    print("Before pruning:", R)
    R = [apply_dominance_pruning(r, ϕ, S) for r in R]
    R = remove_duplicates(R)

    # Final selection
    R = subgroup_selection(R, ϕ, P, S)

    return R[:k]

def calculate_subgroup_quality(subgroup: dict, data: pd.DataFrame) -> int:
    """
    Calculate the quality of a subgroup based on a given measure.
    Favours larger subgroups

    Parameters:
    - subgroup: A dictionary representing the subgroup's conditions.
    - data: The pandas DataFrame containing the dataset.

    Returns:
    - A numerical value representing the quality of the subgroup.
    """
    bonus = 1
    interests = ["cycle_life_0","cycle_life_5","cycle_life_10","cycle_life_15","cycle_life_20","cycle_life_25","cycle_life_30","cycle_life_35","cycle_life_40","cycle_life_45","cycle_life_50","cycle_life_55","cycle_life_60","cycle_life_85","cycle_life_95"]
    filtered_data = data.query(' and '.join([f'{k}=={repr(v)}' if not isinstance(v, str) or '<' not in v and '>' not in v else f'{k}{v}' for k, v in subgroup.items()]))
    for k, _ in subgroup.items():
        if k in interests:
            if bonus == 1:
                bonus = 3
            else:
                bonus = 0
    quality = len(filtered_data) * len(subgroup.keys()) * bonus # Coverage * number of elements * make target columns more valuable
    return quality

# Example usage
# Define the dataset S and quality measure ϕ
S = pd.read_csv("data_binarized.csv") 
S = S.astype('bool')
ϕ = calculate_subgroup_quality  # Quality function
j = 10  # Number of subgroups to mine initially
k = 5   # Number of subgroups to select finally
mincov = 5
maxdepth = 3
P = {'beam_width': 4}

result = dssd(S, ϕ, j, k, mincov, maxdepth, P)
print("Discovered diverse subgroups:", result)
