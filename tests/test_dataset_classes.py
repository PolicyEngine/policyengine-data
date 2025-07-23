"""
Test cases for SingleYearDataset and MultiYearDataset classes.
"""


def test_single_year_dataset() -> None:
    from src.policyengine_data.single_year_dataset import SingleYearDataset
    import pandas as pd

    # Create a sample dataset
    entities = {
        "person": pd.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"]}),
        "household": pd.DataFrame({"id": [1], "income": [50000]}),
    }
    fiscal_year = 2023

    # Initialize SingleYearDataset
    dataset = SingleYearDataset(entities=entities, fiscal_year=fiscal_year)
    dataset.validate()

    # Check if entities are correctly set
    assert len(dataset.entities) == 2
    assert "person" in dataset.entities
    assert "household" in dataset.entities
    pd.testing.assert_frame_equal(
        dataset.entities["person"], entities["person"]
    )
    pd.testing.assert_frame_equal(
        dataset.entities["household"], entities["household"]
    )
    assert dataset.time_period == str(fiscal_year)

    # Save the dataset to a file
    file_path = "test_single_year_dataset.h5"
    dataset.save(file_path)

    loaded_dataset = SingleYearDataset(file_path=file_path)
    loaded_dataset.validate()

    # Check if loaded entities match original entities
    assert len(loaded_dataset.entities) == len(entities)
    for entity_name in entities:
        pd.testing.assert_frame_equal(
            loaded_dataset.entities[entity_name], entities[entity_name]
        )
    assert loaded_dataset.time_period == str(fiscal_year)

    variables = dataset.variables
    assert variables.keys() == {"person", "household"}


def test_multi_year_dataset() -> None:
    from src.policyengine_data.multi_year_dataset import MultiYearDataset
    from src.policyengine_data.single_year_dataset import SingleYearDataset
    import pandas as pd

    # Create SingleYearDataset instances for multiple years
    entities_2023 = {
        "person": pd.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"]}),
        "household": pd.DataFrame({"id": [1], "income": [50000]}),
    }
    entities_2024 = {
        "person": pd.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"]}),
        "household": pd.DataFrame({"id": [1], "income": [55000]}),
    }

    dataset_2023 = SingleYearDataset(entities=entities_2023, fiscal_year=2023)
    dataset_2024 = SingleYearDataset(entities=entities_2024, fiscal_year=2024)

    # Initialize MultiYearDataset with list of SingleYearDataset instances
    multi_dataset = MultiYearDataset(datasets=[dataset_2023, dataset_2024])

    # Check if datasets are correctly set
    assert len(multi_dataset.datasets) == 2
    assert 2023 in multi_dataset.datasets
    assert 2024 in multi_dataset.datasets

    retrieved_2023 = multi_dataset.get_year(2023)
    assert isinstance(retrieved_2023, SingleYearDataset)
    pd.testing.assert_frame_equal(
        retrieved_2023.entities["person"], entities_2023["person"]
    )
    retrieved_2024 = multi_dataset[2024]
    assert isinstance(retrieved_2024, SingleYearDataset)
    pd.testing.assert_frame_equal(
        retrieved_2024.entities["household"], entities_2024["household"]
    )

    # Save the dataset to a file
    file_path = "test_multi_year_dataset.h5"
    multi_dataset.save(file_path)
    loaded_multi_dataset = MultiYearDataset(file_path=file_path)

    # Check if loaded datasets match original datasets
    assert len(loaded_multi_dataset.datasets) == 2
    assert 2023 in loaded_multi_dataset.datasets
    assert 2024 in loaded_multi_dataset.datasets

    for year in [2023, 2024]:
        loaded_year_data = loaded_multi_dataset[year]
        original_year_data = multi_dataset[year]

        for entity_name in original_year_data.entities:
            pd.testing.assert_frame_equal(
                loaded_year_data.entities[entity_name],
                original_year_data.entities[entity_name],
            )

    variables_by_year = multi_dataset.variables
    assert variables_by_year.keys() == {2023, 2024}
    assert ["person", "household"] == list(variables_by_year[2023].keys())
