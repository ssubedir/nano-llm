"""Utility functions for data processing."""

import os
import glob
from typing import List, Union, Optional


def load_text_file(file_path: str) -> List[str]:
    """Load texts from a text file."""
    texts = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if text:
                texts.append(text)
    return texts


def load_text_files(file_paths: Union[str, List[str]]) -> List[str]:
    """
    Load texts from one or more text files.

    Parameters
    ----------
    file_paths : str or List[str]
        Path to a single file or list of file paths

    Returns
    -------
    List[str]
        List of all texts from the files
    """
    if isinstance(file_paths, str):
        file_paths = [file_paths]

    all_texts = []
    for file_path in file_paths:
        if os.path.exists(file_path):
            texts = load_text_file(file_path)
            all_texts.extend(texts)
        else:
            raise FileNotFoundError(f"File not found: {file_path}")

    return all_texts


def get_text_files_from_dir(
    directory: str, recursive: bool = True, pattern: str = "*.txt"
) -> List[str]:
    """
    Get all text files from a directory.

    Parameters
    ----------
    directory : str
        Path to the directory
    recursive : bool, default=True
        Whether to search recursively in subdirectories
    pattern : str, default="*.txt"
        File pattern to match

    Returns
    -------
    List[str]
        List of file paths
    """
    if not os.path.isdir(directory):
        raise NotADirectoryError(f"Directory not found: {directory}")

    if recursive:
        pattern = os.path.join("**", pattern)
        file_paths = glob.glob(os.path.join(directory, pattern), recursive=True)
    else:
        file_paths = glob.glob(os.path.join(directory, pattern))

    return sorted(file_paths)


def load_data_sources(
    data_files: Optional[Union[str, List[str]]] = None,
    data_dir: Optional[str] = None,
    recursive: bool = True,
) -> List[str]:
    """
    Load texts from various data sources (files or directory).

    Parameters
    ----------
    data_files : str or List[str], optional
        Path to one or multiple text files
    data_dir : str, optional
        Path to a directory containing text files
    recursive : bool, default=True
        Whether to search recursively in data_dir

    Returns
    -------
    List[str]
        List of all texts from the specified sources

    Raises
    ------
    ValueError
        If no data source is specified
    FileNotFoundError
        If a specified file or directory doesn't exist
    """
    # Prioritize: data_dir > data_files
    if data_dir:
        file_paths = get_text_files_from_dir(data_dir, recursive=recursive)
        if not file_paths:
            raise ValueError(f"No text files found in directory: {data_dir}")
        return load_text_files(file_paths)
    elif data_files:
        return load_text_files(data_files)
    else:
        raise ValueError("No data source specified. Provide data_files or data_dir.")
